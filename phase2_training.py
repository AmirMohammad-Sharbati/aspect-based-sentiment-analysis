import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler


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
# Assuming 'df' is your pre-loaded Pandas DataFrame
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

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

# ==========================================
# 2. THE RAY TUNE TRAINING FUNCTION
# ==========================================
def train_absa(config):
    # Setup Device (Ray will automatically assign cuda:0 to Trial 1, and cuda:1 to Trial 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2a. Load Tokenizer & Model dynamically based on the current Trial's config
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], 
        num_labels=3
    )
    model.to(device)

    # 2b. Create DataLoaders
    train_dataset = ABSADataset(train_df, tokenizer)
    val_dataset = ABSADataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # 2c. Setup Optimizer, Scheduler, and Mixed Precision Scaler
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # Linear warmup prevents the learning rate from "shocking" the pre-trained model
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp) # MAGIC: Enables 2x speedup on T4 GPUs

    # 2d. The Training Loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # MAGIC: Automatic Mixed Precision (AMP) block
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
            
            # Scaled backprop for numerical stability in FP16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        # 2e. The Validation Loop
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 2f. Calculate MACRO F1 (Crucial for Imbalanced SemEval data)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)

        # 2g. Report to Ray Tune (This triggers ASHA to keep or kill the trial)
        train.report({"val_loss": avg_val_loss, "macro_f1": macro_f1})

# ==========================================
# 3. CONFIGURE AND LAUNCH THE SWEEP
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    # Define what we want to test
    search_space = {
        "model_name": tune.choice(["roberta-base", "microsoft/deberta-v3-base"]),
        "lr": tune.loguniform(1e-5, 5e-5),
        "batch_size": tune.choice([16, 32]),
        "weight_decay": tune.choice([0.01, 0.1]),
        "epochs": 4 # Maximum epochs a model is allowed to train
    }

    # Setup the "Time Saver" Scheduler
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='macro_f1',
        mode='max',
        max_t=4,        # Max epochs
        grace_period=1, # Look at models after 1 epoch before killing the bad ones
        reduction_factor=2
    )

    # MAGIC: Tell Ray we have 2 GPUs, give 1 to each trial so 2 run at the same time
    tuner = tune.Tuner(
        tune.with_resources(train_absa, resources={"gpu": 1, "cpu": 2}),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=20, scheduler=asha_scheduler),
    )

    print("Starting 20-model audition across 2x T4 GPUs...")
    results = tuner.fit()

    # ==========================================
    # 4. GET THE BEST RESULT
    # ==========================================
    top_results = results.get_best_results(metric="macro_f1", mode="max", num_results=3)

    print("\nTop 3 Configurations:")
    for i, res in enumerate(top_results):
        print(f"\nRank {i+1}")
        print("Config:", res.config)
        print("Validation Macro F1:", res.metrics["macro_f1"])

    best_config = top_results[0].config



    print("\nRetraining best configuration on train+val...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine train and val for final training
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(best_config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        best_config["model_name"],
        num_labels=3
    ).to(device)

    full_dataset = ABSADataset(full_df, tokenizer)
    full_loader = DataLoader(full_dataset, batch_size=best_config["batch_size"], shuffle=True)

    optimizer = AdamW(
        model.parameters(),
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    total_steps = len(full_loader) * best_config["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(best_config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in full_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(full_loader):.4f}")

    model.save_pretrained("./best_final_model")
    tokenizer.save_pretrained("./best_final_model")
    print("Final model saved.")