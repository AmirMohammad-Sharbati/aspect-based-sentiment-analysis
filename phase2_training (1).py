import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import json


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

# ==========================================
# 1. DATA PREPARATION (Executes Once)
# ==========================================

class ABSADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row["sentence"],
            row["aspect"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }


def train_absa(config):
    # FIX 1: Safely get data from Ray's shared memory
    train_df = ray.get(config["train_data_ref"])
    val_df = ray.get(config["val_data_ref"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=3).to(device)

    train_loader = DataLoader(ABSADataset(train_df, tokenizer), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(ABSADataset(val_df, tokenizer), batch_size=config["batch_size"])

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        v_loss, v_correct, v_total, all_preds, all_labels = 0, 0, 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=mask)
                loss = criterion(outputs.logits, labels)
                v_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

        # FIX 2: Report ALL metrics for plotting
        tune.report({
            "val_loss": v_loss / len(val_loader),
            "train_loss": total_loss / len(train_loader),
            "val_acc": v_correct / v_total,
            "train_acc": correct / total,
            "macro_f1": f1_score(all_labels, all_preds, average='macro')
        })

if __name__ == "__main__":
    set_seed(42)
    ray.init(ignore_reinit_error=True) # FIX 3: Prevent Kaggle crashes
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_ref = ray.put(train_df)
    val_ref = ray.put(val_df)

    search_space = {
        "model_name": tune.choice(["roberta-base", "microsoft/deberta-v3-base"]),
        "lr": tune.loguniform(1e-5, 5e-5),
        "batch_size": tune.choice([16, 32]),
        "weight_decay": tune.choice([0.01, 0.1]),
        "epochs": 4,
        "train_data_ref": train_ref,
        "val_data_ref": val_ref
    }

    tuner = tune.Tuner(
        tune.with_resources(train_absa, resources={"gpu": 1, "cpu": 2}),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=20, scheduler=ASHAScheduler(metric='macro_f1', mode='max', max_t=4)),
    )
    results = tuner.fit()

    # Save ALL experiment results to CSV (VERY IMPORTANT FOR REPORT)
    all_results = results.get_dataframe()
    all_results.to_csv("all_models_results.csv", index=False)
    print("All model results saved to all_models_results.csv")

    # Plotting the Best Model
    best_res = results.get_best_result(metric="macro_f1", mode="max")
    pdf = best_res.metrics_dataframe.sort_values("training_iteration")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(pdf['train_loss'], label='Train Loss'); plt.plot(pdf['val_loss'], label='Val Loss')
    plt.title('Loss History'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(pdf['val_acc'], label='Val Acc'); plt.plot(pdf['macro_f1'], label='Macro F1')
    plt.title('Metrics History'); plt.legend()
    plt.show()
    
    # ==========================================
    # 4. RESULTS & SUMMARY
    # ==========================================
    top_results = results.get_best_results(metric="macro_f1", mode="max", num_results=3)

    print("\n Top 3 Configurations Summary:")
    for i, res in enumerate(top_results):
        print(f"Rank {i+1}: {res.config['model_name']} | F1: {res.metrics['macro_f1']:.4f} | LR: {res.config['lr']}")

    best_config = best_res.config # Use the result from our plotting section

    # Retraining logic
    print("\n Retraining final model on 100% of data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(best_config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(best_config["model_name"], num_labels=3).to(device)

    full_loader = DataLoader(ABSADataset(full_df, tokenizer), batch_size=best_config["batch_size"], shuffle=True)
    optimizer = AdamW(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
    
    total_steps = len(full_loader) * best_config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(best_config["epochs"]):
        model.train()
        total_loss = 0
        for batch in full_loader:
            ids, mask, lbls = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask).logits, lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Final Retrain - Epoch {epoch+1} Loss: {total_loss/len(full_loader):.4f}")

    # SAVE EVERYTHING
    model.save_pretrained("./best_final_model")
    tokenizer.save_pretrained("./best_final_model")
    
    # Clean config for JSON saving
    saveable_config = {k: v for k, v in best_config.items() if not k.endswith('_ref')}
    with open("best_config.json", "w") as f:
        json.dump(saveable_config, f, indent=4)

    print("\n DONE! Download 'all_models_results.csv' and the 'best_final_model' folder for your report.")
