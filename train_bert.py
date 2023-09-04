import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from bert_torch_dataset_creator import GPTweetDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from Bert_SentimentClassifier import SentimentClassifier
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.ERROR)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

#Constants
RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_LEN = 100
class_names = ['negative', 'neutru','positive']

#Read and explore the data a bit
pd.set_option('display.max_columns', None)
#df = pd.read_csv("Data.csv")
#print(df.head())

#load pre-trained model
PRE_TRAINED_MODEL_NAME = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Let's split the data:
#df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

df_train = pd.read_csv(r"train2.csv", encoding='utf-8')
df_test = pd.read_csv(r"test2.csv", encoding='utf-8')
df_val = pd.read_csv(r"val2.csv", encoding='utf-8')
#df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print(df_train.shape, df_val.shape, df_test.shape)

#Create a data loader
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
    num_workers=0
  )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
print(type(train_data_loader))
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


#bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Create an instance bert model and move it to gpu
model = SentimentClassifier(len(class_names))
model = model.to(device)

#We'll move the example batch of our training data to the GPU:
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

#To get the predicted probabilities from our trained model, we'll apply the softmax function to the outputs:
F.softmax(model(input_ids, attention_mask), dim=1)

EPOCHS = 5

#Training using AdamW optimizer provided by Hugging Face
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
#Loss function is CrossEntropy
loss_fn = nn.CrossEntropyLoss().to(device)

#Helper function to train th emodel for one epoch
def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

#Training the model should look familiar, except for two things. The scheduler gets called every time a batch is fed to the model. We're avoiding exploding gradients by clipping the gradients of the model using clip_grad_norm_.
#Let's write another one that helps us evaluate the model on a given data loader:

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

    #Saving to history
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
      #Save the model
    torch.save(model.state_dict(), r'SA3_march2022.bin')
    best_accuracy = val_acc


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.show()