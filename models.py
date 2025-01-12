import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from transformers import pipeline
import json
import streamlit as st

def prepare_data():
    df=pd.read_csv('./cleaned_data/cleaned_rev.csv',delimiter=',')
    df = df['note','avis_cor_en']
    x_train, x_test, y_train, y_test = train_test_split(df['avis_cor_en'], df['note'], test_size=0.2)
    
    train_data = Dataset.from_dict({
    'text': x_train.tolist(),
    'label': y_train.tolist()
    })
    test_data = Dataset.from_dict({
        'text': x_test.tolist(),
        'label': y_test.tolist()
    })
    return train_data, test_data

def model_setup():
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    
    train_data, test_data = prepare_data()
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    )

    trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_data,            # training dataset
    eval_dataset=test_data               # evaluation dataset
    )

    # Train the model
    trainer.train()

@st.cache_resource
def load_model_and_tokenizer_star():
    model_path = "./model_saved"  # Replace with the actual path
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def process_input_star(input_rev):
    model, tokenizer = load_model_and_tokenizer_star()

    inputs = tokenizer(
        input_rev,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def call_zero_shot_topic(input_rev):
    with open('./cleaned_data/topics_word.json', 'r') as f:
        topics_word = json.load(f)

    with open('./cleaned_data/topics_name.json', 'r') as f:
        topics_name = json.load(f)

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = list(topics_name.values())
    result = classifier(input_rev, candidate_labels=labels)
    return result
