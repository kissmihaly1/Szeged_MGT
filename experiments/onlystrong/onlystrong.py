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

#data
log_wandb = True
if log_wandb:
    wandb.login(key='-')
train_df = pd.read_json('-', lines= True)
dev_df = pd.read_json('-', lines = True)


group_0= ['human']

group_1 = [
'flan_t5_small',
'opt_125m',
'flan_t5_base',
'opt_350m',
'flan_t5_large',
'opt_1.3b',
'opt_iml_max_1.3b',
'opt_2.7b',
't0_3b',
'flan_t5_xl',
't0_11b',
'flan_t5_xxl',
'opt_13b',
'gpt_neox',
'opt_iml_30b',
'opt_30b',
'dolly',
'gpt_j',
'opt_6.7b',
'bloomz',
'bloom_7b',
]

group_2 = [
'davinci',
'gpt-35',
'text-davinci-003',
'text-davinci-002',
'gpt-3.5-turbo',
'cohere',
'gpt4o',
'gpt4',
'GLM130B',
'mixtral-8x7b',
'gemma-7b-it',
'7B',
'llama3-8b',
'gemma2-9b-it',
'65B',
'30B',
'13B',
'llama3-70b',
'dolly-v2-12b',
]

model_to_group = {model: 0 for model in group_0}
model_to_group.update({model: 2 for model in group_1})
model_to_group.update({model: 1 for model in group_2})


train_df['group_label'] = train_df['model'].map(model_to_group)
dev_df['group_label'] = dev_df['model'].map(model_to_group)

train_df = train_df[train_df['group_label'] != 2]

dev_df = dev_df[dev_df['group_label'] != 2]
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



human_label = 0


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['group_label'].tolist()
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


for model_index in range(3):
    print(f"Training model {model_index + 1}/3")

    best_f1 = 0
    best_f1_overall = 0
    if model_index == 0:
        seed = 33
        set_seed(seed)
        if log_wandb:
            wandb.init(project="-", name=f"deberta_run_{model_index + 3}_gpu1_2group_seed{seed}", reinit=True)
    elif model_index == 1:
        seed = 22
        set_seed(seed)
        if log_wandb:
            wandb.init(project="-", name=f"deberta_run_{model_index + 1}_gpu1_2group_seed{seed}", reinit=True)

    elif model_index == 2:
        seed = 11
        set_seed(seed)
        if log_wandb:
            wandb.init(project="-", name=f"deberta_run_{model_index + 1}_gpu1_2group_seed{seed}", reinit=True)


    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)


    val_number_per_epoch = 3
    early_stopping_patience = 3
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
        for step, batch in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

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
                    best_model_path = f"Model_groups_onlystrong_seed{seed}"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Model saved with f1: {f1}")

                print("\n------------------------------")


        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

        if log_wandb:
            wandb.log({"model_index": model_index + 1, "epoch": epoch + 1, "train_loss": avg_train_loss, "learning_rate": scheduler.get_last_lr()[0]})

    end_time = time.time()
    training_time = end_time - start_time
    if log_wandb:
        wandb.log({"model_index": model_index + 1, "training_time_seconds": training_time})


    best_model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)
    best_model.eval()

    dev_df['softmax_logits'] = None
    softmax_logits_list = []
    all_preds_thr = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dev_loader, desc='Testing on dev set')):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Base binary test accuracy: {accuracy:.4f}, Test F1 score: {f1:.4f}')

    if log_wandb:
        wandb.log({"test_accuracy": accuracy, "test_f1": f1})

    if log_wandb:
        wandb.finish()
