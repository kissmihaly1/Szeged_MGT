import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import wandb
import os
import time
import random

log_wandb = False
if log_wandb:
    wandb.login(key='-')
train_df = pd.read_json('-', lines= True)
dev_df = pd.read_json('-', lines = True)

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, num_workers=2, pin_memory=True)

val_steps_per_epoch = 2

val_steps = len(train_loader)//val_steps_per_epoch


epochs = 2
accumulation_steps = 4
best_f1_overall = 0


for model_index in range(1):
    print(f"Training model {model_index + 1}/2")
    if model_index == 0:
        seed = 22
        set_seed(22)
    if model_index == 1:
        seed = 10
        set_seed(10)

    if log_wandb:
        wandb.init(project="-", name=f"deberta_run_hop_model_{model_index + 1}", reinit=True)


    model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)


    val_number_per_epoch=4
    early_stopping_patience = 2
    best_f1 = 0
    patience_counter = 0
    start_time = time.time()
    val_number = epochs*val_number_per_epoch
    val_count = 0
    val_steps = len(train_loader)//val_number_per_epoch

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch + 1}/{epochs}')):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            #validation by steps
            if (step + 1) % val_steps == 0:
                val_count += 1
                model.eval()
                total_val_loss = 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc=f'Validation {val_count}/{val_number}', position=0, leave=True):
                        input_ids, attention_mask, labels = [x.to(device) for x in batch]
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        val_loss = outputs.loss
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(dev_loader)
                accuracy = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='macro')
                print(f'\n {val_count}/{val_number}, val_accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, val_loss: {avg_val_loss:.4f}')

                if log_wandb:
                    wandb.log({"model_index": model_index + 1, "val_count": val_count, "val_accuracy": accuracy, "val_f1": f1})


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
                    best_model_path = f"Model_{model_index}_{seed}"
                    torch.save(model.state_dict(), best_model_path)



        avg_train_loss = total_loss / len(train_loader)
        print(f'epoch {epoch + 1}/{epochs}, train_loss: {avg_train_loss:.4f}')

        if log_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "learning_rate": scheduler.get_last_lr()[0]})




    end_time = time.time()
    training_time = end_time - start_time

    if log_wandb:
        wandb.log({"model_index": model_index + 1, "training_time_seconds": training_time})

if log_wandb:
    wandb.finish()
