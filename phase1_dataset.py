import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import re


print(plt.style.available)
plt.style.use('dark_background')
# In kaggle
# train_xml_path = "/kaggle/input/datasets/amirmohammadsharbati/dataset/Restaurants_Train_v2.xml"
train_xml_path = "./data/Restaurants_Train_v2.xml"


tree = ET.parse(train_xml_path)
root = tree.getroot()

data = []

for sentence in root.iter("sentence"):
    text_elem = sentence.find("text")
    if text_elem is None:
        continue
    text = text_elem.text
    
    aspect_terms = sentence.find("aspectTerms")
    if aspect_terms is not None:
        for aspect in aspect_terms.findall("aspectTerm"):
            term = aspect.get("term")
            polarity = aspect.get("polarity")
            if polarity != "conflict":
                data.append({
                    "sentence": text,
                    "aspect": term,
                    "polarity": polarity
                })

df = pd.DataFrame(data)
df.head()

print("total number of datas: ", len(df))
print("\ndistro")
print(df["polarity"].value_counts())


plt.figure()
sns.countplot(x="polarity", data=df)
plt.title("Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


stopwords = set(STOPWORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

df["clean_sentence"] = df["sentence"].apply(clean_text)
df["word_count"] = df["sentence"].apply(lambda x: len(x.split()))
df[["sentence", "word_count"]].head()


plt.figure()
plt.hist(df["word_count"], bins=30)
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


# Function to generate and show wordcloud
def show_wordcloud(text, title):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        max_words=100,
        random_state=42
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Generate texts
pos_text = " ".join(df[df["polarity"]=="positive"]["clean_sentence"])
neg_text = " ".join(df[df["polarity"]=="negative"]["clean_sentence"])
neu_text = " ".join(df[df["polarity"]=="neutral"]["clean_sentence"])

# Plot
show_wordcloud(pos_text, "Positive Sentences Word Cloud")
show_wordcloud(neg_text, "Negative Sentences Word Cloud")
show_wordcloud(neu_text, "Neutral Sentences Word Cloud")