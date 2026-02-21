import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = "./best_final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

def parse_semeval_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    sentences = []
    aspects = []
    labels = []

    polarity_map = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
    }

    for sentence in root.findall("sentence"):
        text = sentence.find("text").text
        
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is None:
            continue
        
        for aspect in aspect_terms.findall("aspectTerm"):
            polarity = aspect.attrib["polarity"]
            if polarity == "conflict":
                continue  # معمولا حذف می‌شود
            
            sentences.append(text)
            aspects.append(aspect.attrib["term"])
            labels.append(polarity_map[polarity])

    return pd.DataFrame({
        "sentence": sentences,
        "aspect": aspects,
        "label": labels
    })


test_df = parse_semeval_xml("./data/Restaurants_Test_Data_phaseB.xml")
print("Test size:", len(test_df))
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


test_dataset = ABSADataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32)


all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:\n")

target_names = ["Negative", "Neutral", "Positive"]

print(classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    digits=4
))

macro_f1 = f1_score(all_labels, all_preds, average="macro")
print("Macro-F1:", macro_f1)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Restaurants PhaseB")
plt.show()

error_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]

print("\nSample Errors:\n")

for idx in error_indices[:5]:
    print("Sentence:", test_df.iloc[idx]["sentence"])
    print("Aspect:", test_df.iloc[idx]["aspect"])
    print("True Label:", target_names[all_labels[idx]])
    print("Predicted:", target_names[all_preds[idx]])
    print("-"*60)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef
)

target_names = ["Negative", "Neutral", "Positive"]

print("\n=== Detailed Classification Report ===\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    digits=4
))

accuracy = accuracy_score(all_labels, all_preds)

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro"
)

weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted"
)

kappa = cohen_kappa_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print("\n=== Global Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall: {macro_r:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=target_names,
    output_dict=True
)

precision_vals = [report_dict[c]["precision"] for c in target_names]
recall_vals = [report_dict[c]["recall"] for c in target_names]
f1_vals = [report_dict[c]["f1-score"] for c in target_names]

x = np.arange(len(target_names))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision_vals, width, label="Precision")
plt.bar(x, recall_vals, width, label="Recall")
plt.bar(x + width, f1_vals, width, label="F1-score")

plt.xticks(x, target_names)
plt.ylim(0, 1)
plt.legend()
plt.title("Per-Class Performance Comparison")
plt.show()

true_counts = np.bincount(all_labels)
pred_counts = np.bincount(all_preds)

x = np.arange(len(target_names))

plt.figure(figsize=(8,5))
plt.bar(x - 0.2, true_counts, width=0.4, label="True")
plt.bar(x + 0.2, pred_counts, width=0.4, label="Predicted")

plt.xticks(x, target_names)
plt.legend()
plt.title("True vs Predicted Distribution")
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

all_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)

y_true_bin = label_binarize(all_labels, classes=[0,1,2])

plt.figure(figsize=(7,6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{target_names[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()