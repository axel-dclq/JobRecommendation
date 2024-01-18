import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Étape 1: Prétraitement des données
data_user = pd.read_csv('data\survey_results_public.csv')

skills_col = ['LanguageWorkedWith','DatabaseWorkedWith',
              'PlatformWorkedWith','FrameworkWorkedWith',
              'IDE','OperatingSystem']

data_user_skills = data_user[['Respondent', 'DevType', 'YearsCoding'] + skills_col]
data_user_skills.fillna('', inplace=True)
data_user_skills['skills'] = (data_user_skills['LanguageWorkedWith'].apply(lambda x: ' '.join(x.split(';')) + ' ') + 
                              data_user_skills['DatabaseWorkedWith'].apply(lambda x: ' '.join(x.split(';')) + ' ') + 
                              data_user_skills['PlatformWorkedWith'].apply(lambda x: ' '.join(x.split(';')) + ' ') + 
                              data_user_skills['FrameworkWorkedWith'].apply(lambda x: ' '.join(x.split(';')) + ' ') + 
                              data_user_skills['IDE'].apply(lambda x: ' '.join(x.split(';')) + ' ') + 
                              data_user_skills['OperatingSystem'].apply(lambda x: ' '.join(x.split(';')))).str.strip()

data_user_skills.drop(columns=['LanguageWorkedWith','DatabaseWorkedWith',
                                'PlatformWorkedWith','FrameworkWorkedWith',
                                'IDE','OperatingSystem'],
                        axis=1, 
                        inplace=True)

data_user_skills.drop(data_user_skills[data_user_skills['skills'] == ''].index, inplace=True)

texts = data_user_skills['skills'].to_list()

label_encoder = LabelEncoder()
labels = data_user_skills['DevType'].to_list()
labels = label_encoder.fit_transform(labels)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    
def train(model, data_loader, optimizer, scheduler, device):
    # Set the model to training mode
    model.train()

    # Iterate over mini-batches in the training data
    for batch in data_loader:
        # Zero the gradients to avoid accumulation
        optimizer.zero_grad()

        # Move input data to the specified device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass: compute model predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate the CrossEntropyLoss between predictions and true labels
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()

        # Update model parameters using the optimizer
        optimizer.step()

        # Adjust the learning rate using the scheduler
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def predict_class(text, model, tokenizer, device, label_encoder, max_length=128):
    model.eval()
    
    # Tokenization
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    # Convertir l'indice de classe prédite au nom de la classe
    predicted_class_name = label_encoder.inverse_transform([preds.item()])[0]

    return predicted_class_name


# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 21
max_length = 40
batch_size = 100
num_epochs = 4
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)