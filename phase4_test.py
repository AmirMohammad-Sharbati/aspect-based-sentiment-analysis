import torch

# Define the sentiment mapping for the report
target_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_custom_sentences(file_path, model, tokenizer):
    model.eval()
    results = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    print(f"{'Sentence':<50} | {'Aspect':<15} | {'Prediction'}")
    print("-" * 80)
    
    for line in lines:
        if "|" not in line: continue # Skip malformed lines
        
        text, aspect = line.strip().split("|")
        
        # Tokenize pair
        inputs = tokenizer(
            text.strip(), 
            aspect.strip(), 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            prediction = target_names[pred_idx]
            
        print(f"{text.strip()[:47]+'...':<50} | {aspect.strip():<15} | {prediction}")
        results.append((text, aspect, prediction))
    
    return results

# Run it
custom_results = predict_custom_sentences("difficult_sentences.txt", model, tokenizer)




########################################### second way
difficult_test_cases = [
    ("The meal tasted great, but the waiter’s attitude ruined the evening.", "waiter"),
    ("Prices are reasonable — unless you order the specials, then it gets expensive.", "specials"),
    ("The place is cozy, though seating is limited during peak hours.", "seating"),
    ("I loved the main course; dessert, however, tasted like it came straight from a freezer.", "dessert"),
    ("They took our reservation but the table was sticky when we sat down.", "table"),
    ("Perfect ambience for dates — at least until the kids started running around.", "ambience"),
    ("I thought the food was good, but I’d hesitate to call it healthy.", "food"),
    ("We waited half an hour for our order — not sure if the kitchen or the app was to blame.", "order"),
    ("It’s cheaper than competitors, yet it feels less sturdy in quality.", "quality"),
    ("Great flavors — until you bite into a chunk of undercooked meat.", "meat"),
    ("The menu photos are vibrant, though they didn’t match what arrived.", "menu photos"),
    ("They claim 24/7 delivery; I called at 2 AM and no one picked up.", "delivery"),
    ("It felt like a steal during the sale, but I doubt it’s worth full price.", "price"),
    ("The instructions for dietary requests were confusing, yet the chef managed to accommodate us.", "chef"),
    ("Although not the priciest option, I regret ordering it.", "ordering"),
    ("An improvement from their last location, but still not as good as I remember.", "location"),
    ("Handles rush hour fine, but the AC fails when the crowd grows.", "AC"),
    ("Nice packaging for takeaway, but the seal was broken and the soup spilled.", "seal"),
    ("The app launches instantly — except when it forces an update and stalls.", "app"),
    ("I’d recommend it for families, but not if you’re looking for fine dining.", "fine dining")
]

model.eval()
target_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"{'Aspect':<15} | {'Prediction':<10} | {'Sentence'}")
print("-" * 100)

for text, aspect in difficult_test_cases:
    # Prepare inputs for the model
    inputs = tokenizer(
        text, 
        aspect, 
        truncation=True, 
        padding="max_length", 
        max_length=128, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        prediction = target_names[pred_idx]

    print(f"{aspect:<15} | {prediction:<10} | {text}")








comparison_pairs = [
    # Format: (Sentence A, Sentence B, Target Aspect)
    ("The food was delicious, but the service was painfully slow.", "The food was delicious — the service, however, was painfully slow.", "service"),
    ("I loved the ambience; the music made the evening perfect.", "I loved the ambience (the music made the evening perfect).", "ambience"),
    ("They honored my dietary request and the dish was great.", "They claim to have honored my dietary request and the dish was great.", "dietary request"),
    ("Delivery arrived cold, but the packaging looked very professional.", "Packaging looked very professional — delivery arrived cold.", "delivery"),
    ("The menu pictures were accurate; what I got matched perfectly.", "The menu pictures weren’t inaccurate; what I got matched perfectly.", "menu pictures"),
    ("Wait time was 45 minutes — unacceptable for lunch.", "Wait time was 45 minutes — not exactly ideal for lunch.", "wait time"),
    ("The server was rude and ignored our questions.", "The server, who ignored our questions, was not exactly friendly.", "server"),
    ("Food portion was generous; value for money is excellent.", "Food portion was generous; therefore value for money is not bad.", "value for money"),
    ("The soup was undercooked and tasted raw.", "The soup tasted kind of raw — undercooked, really.", "soup"),
    ("They offer quick takeout service which I appreciate.", "Quick takeout service — I appreciated that.", "takeout service"),
    ("I’d recommend this place to families; it’s kid-friendly.", "I’d recommend this place to families — maybe not for couples seeking quiet.", "families"),
    ("The portion sizes were small, yet the flavors were intense.", "Despite small portion sizes, the flavors were intense.", "flavors"),
    ("The staff handled my allergy carefully.", "The staff ‘handled’ my allergy carefully.", "staff"),
    ("The dessert looked beautiful but tasted bland.", "Beautiful-looking dessert; taste? Bland.", "dessert"),
    ("Service was good when we arrived, but declined as the night went on.", "Service was good at first; later service declined.", "service"),
    ("App checkout froze during peak hours.", "App checkout froze — only during peak hours though.", "app checkout"),
    ("The hostess was polite but seated us near the kitchen.", "Polite hostess, seated us near the kitchen.", "hostess"),
    ("Menu photos are vibrant, though they didn’t match what arrived.", "Menu photos are vibrant — they didn’t match what arrived.", "menu photos"),
    ("I appreciated that they refilled water often.", "Refilled water often — appreciated that.", "water"),
    ("The steak was perfectly cooked; no complaints.", "Perfectly cooked steak — no complaints at all.", "steak")
]


model.eval()
target_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"{'Aspect':<15} | {'Pred A':<10} | {'Pred B':<10} | {'Change?'}")
print("-" * 60)

for sent_a, sent_b, aspect in comparison_pairs:
    preds = []
    for text in [sent_a, sent_b]:
        inputs = tokenizer(text, aspect, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds.append(torch.argmax(outputs.logits, dim=1).item())
    
    label_a = target_names[preds[0]]
    label_b = target_names[preds[1]]
    changed = "YES" if preds[0] != preds[1] else "no"
    
    print(f"{aspect:<15} | {label_a:<10} | {label_b:<10} | {changed}")