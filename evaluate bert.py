from Bert_SentimentClassifier import SentimentClassifier
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from bert_torch_dataset_creator import GPTweetDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import seaborn as sn
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
from sklearn.metrics import classification_report
import itertools
import logging
logging.basicConfig(level=logging.ERROR)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
BATCH_SIZE = 8
MAX_LEN = 512

#denumirea claselor in ordine alfabetica
class_names = ['negative', 'neutru','positive']

PRE_TRAINED_MODEL_NAME = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#deschid modelul
model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load(r'SA4_march2022_512_2ndtry.bin', map_location=torch.device('cpu')))
model = model.to(device)

# deschid datele de test
df_test = pd.read_csv("test2.csv")

#label_encoder = LabelEncoder()
#df_test.Label = label_encoder.fit_transform(df_test.Label)

#creez data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPTweetDataset(
    texts=df.text.to_numpy(),
    targets=df.Label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,

  )
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)



def get_predictions(model, data_loader):
    # function to get predictions
    model = model.eval()

    question_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["tweet_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            # probabilities
            probs = F.softmax(outputs, dim=1)

            question_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return question_texts, predictions, prediction_probs, real_values


#tweet-urile din datele de test
texts_test = df_test.text.tolist()
#labelurile tweet-urilor de test
labels_test = df_test.Label

# bert_predictions are th epredicted labels
bert_predictions = get_predictions(model, test_data_loader)
bert_predictions = bert_predictions[1].tolist()
print(len(bert_predictions), bert_predictions)

# labels_test is a series containing the true labels
true = labels_test.values.tolist()
print(len(true), true)



#matricea de confuzie
cm = confusion_matrix(true, bert_predictions, labels=[0,1,2])
print(cm)

#matricea de confuzie normalizata
cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm2)

df_cm = pd.DataFrame(cm2, index = [i for i in class_names], columns = [i for i in class_names])
sn.set(font_scale=1.4) # for label size
#add fmt = 'd' for printing cm
sn.heatmap(df_cm, annot=True, cmap="Blues") #annot_kws={"size": 16}
plt.show()

from sklearn.metrics import classification_report
print(classification_report(true, bert_predictions, target_names=class_names))

