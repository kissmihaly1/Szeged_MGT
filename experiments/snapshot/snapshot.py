import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import wandb
import os
import time
import torch.nn.functional as F
import random
import math

#data
log_wandb = True
if log_wandb:
    wandb.login(key='-')
train_df = pd.read_json('/home/kissmihaly/mgt/data/train.jsonl', lines= True)
dev_df = pd.read_json('/home/kissmihaly/mgt/data/dev.jsonl', lines = True)


tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


#Cosine annealing learning rate
class CosineAnnealingLR:
    def __init__(self, total_steps, base_lr=1e-4, min_lr=1e-5, decay_factor=0.95):
        """
        Parameters:
        - base_lr: The initial learning rate (used during validation and reset).
        - min_lr: The minimum learning rate to reach during teaching.
        - total_steps: The number of steps over which the cosine annealing is applied.
        - decay_factor: The factor by which base_lr and min_lr decrease after each reset.
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.lr = self.base_lr
        self.val_count = 0
        self.decay_factor = decay_factor

    def get_lr(self):
        """
        Returns the current learning rate.
        """
        return self.lr

    def decrease_lr(self):
        """
        Decreases the learning rate based on the cosine annealing schedule.
        """
        cos_inner = math.pi * self.current_step / self.total_steps
        self.lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(cos_inner)) / 2
        self.current_step += 1
        return self.lr

    def reset(self):
        """
        Resets the scheduler for the next teaching phase.
        """
        self.val_count += 1
        self.current_step = 0
        self.base_lr *= self.decay_factor
        self.min_lr *= self.decay_factor




#encoding
label_encoder = LabelEncoder()
train_df['encoded_model'] = label_encoder.fit_transform(train_df['model'])
dev_df['encoded_model'] = dev_df['model'].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
human_label = label_encoder.transform(['human'])[0]


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['encoded_model'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

train_dataset = TextDataset(train_df, tokenizer)
dev_dataset = TextDataset(dev_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, num_workers=2, pin_memory=True)


epochs = 3
accumulation_steps = 4
best_f1_overall = 0
best_model_path = "best_model.pth"

#models
for model_index in range(1):
    print(f"Training model {model_index + 1}/5")

    if log_wandb:
        wandb.init(project="-", name=f"deberta_snap_run_gpu3", reinit=True)


    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=41)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    val_number_per_epoch=2
    early_stopping_patience = 2
    best_f1 = 0
    patience_counter = 0
    val_number = epochs*val_number_per_epoch
    val_count = 0
    val_steps = len(train_loader)//val_number_per_epoch

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingLR(total_steps=val_steps)



    start_time = time.time()

    #training
    for epoch in range(epochs):
        model_path = f"model_val_{val_count}"
        dev_df[model_path] = None
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            lr = scheduler.decrease_lr()
            wandb.log({"learning_rate": lr})
            total_loss += loss.item() * accumulation_steps

            if (step + 1) % val_steps == 0:
                val_count += 1
                model.eval()
                total_val_loss = 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(tqdm(dev_loader, desc=f'Validation {val_count}/{val_number}', position=0, leave=True)):
                        input_ids, attention_mask, labels = [x.to(device) for x in batch]
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        val_loss = outputs.loss
                        total_val_loss += val_loss.item()
                        probabilities = F.softmax(logits, dim=1)

                        for i, prob in enumerate(probabilities.cpu().numpy()):
                            dev_df.at[batch_idx * dev_loader.batch_size + i, model_path] = prob.tolist()

                avg_val_loss = total_val_loss / len(dev_loader)
                accuracy = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='macro')
                print(f'\n {val_count}/{val_number}, val_accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, val_loss: {avg_val_loss:.4f}')

                if log_wandb:
                    wandb.log({"model_index": model_index + 1, "val_count": val_count, "val_accuracy": accuracy, "val_f1": f1})
                scheduler.reset()

                if f1 >= best_f1:
                    best_f1 = f1
                    patience_counter = 0

                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping")
                        break

                if best_f1 > best_f1_overall:
                    best_f1_overall = best_f1

                best_model_path = f"/home/kissmihaly/mgt/models/model_val_{val_count}"
                torch.save(model.state_dict(), best_model_path)
                print("\n------------------------------")


        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

        if log_wandb:
            wandb.log({"model_index": model_index + 1, "epoch": epoch + 1, "train_loss": avg_train_loss})

    end_time = time.time()
    training_time = end_time - start_time
    if log_wandb:
        wandb.log({"model_index": model_index + 1, "training_time_seconds": training_time})

    dev_df.to_csv('dev_logits.csv', index=False)
