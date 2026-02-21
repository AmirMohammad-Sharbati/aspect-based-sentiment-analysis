model.eval()


import torch

y_true = []
y_pred = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["positive","negative","neutral"],
            yticklabels=["positive","negative","neutral"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



# ==============================
# Error Analysis
# ==============================

print("\n--- Error Analysis Example ---\n")

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        
        # Get original sentence and aspect from validation dataframe
        sentence = val_df.iloc[i]["sentence"]
        aspect = val_df.iloc[i]["aspect"]
        
        # Convert numeric labels back to text
        id2label = {0: "positive", 1: "negative", 2: "neutral"}
        
        print("Sentence:", sentence)
        print("Aspect:", aspect)
        print("True Label:", id2label[y_true[i]])
        print("Predicted Label:", id2label[y_pred[i]])
        print("\nPossible Reason for Error:")
        print("- The sentence may contain multiple sentiment expressions.")
        print("- The model may have focused on the wrong opinion word.")
        print("- Neutral class is harder and underrepresented.")
        break