import argparse
import constant as config
import torch
from util.dataset import read_dataset, sampling_dataset, Dataset
from util.model import BERTClass
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from trainer import Trainer
# from util.augment import *
import numpy as np
import torch.nn.functional as F 
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# Define the dataset path and class number
# TODO: change here to load different datasets.
# Define the dataset path and class number
labeled_dataset_path = "./Dataset/Labeled_new.csv"
test_dataset_path = "./Dataset/Val.csv"
unlabeled_dataset_path = "./Dataset/Unlabeled.csv"
save_path = "./exaperimental_results/"

if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Load the dataset
test_df = pd.read_csv(test_dataset_path)
labeled_df = pd.read_csv(labeled_dataset_path)
unlabled_df = pd.read_csv(unlabeled_dataset_path)

test_df = test_df[['comments', 'Helpful']]
labeled_df = labeled_df[['comments', 'Helpful']]
unlabled_df = unlabled_df[['comments']]

test_df["Helpful"] = test_df["Helpful"].apply(lambda x: 1 if x == 1 else 0)

#Make it balanced if needed
# labeled_df = labeled_df.groupby('Helpful', as_index=False).apply(lambda x: x.sample(7000))

print("Count of labelled data - ", labeled_df['Helpful'].value_counts())
print("Count of test data - ", test_df['Helpful'].value_counts())
input("Press Y to continue, or ctrl+c to exit: ")

# # Load the dataset
# train_df = pd.read_csv(Labeled_dataset_path)
# unlabeled_df = pd.read_csv(Unlabeled_dataset_path)
# unlabeled_texts, unlabeled_labels = unlabeled_df["comments"].tolist(),[0]*len(unlabeled_df["comments"])

# # Modifiy the labels from -1 to 0 since in bert model, target should contain indices in the range [0, nb_classes-1].
# train_df["Helpful"] = train_df["Helpful"].apply(lambda x: 1 if x == 1 else 0)


# Define the training and validation sets
labeled_texts, labeled_labels = labeled_df["comments"].tolist(), labeled_df["Helpful"].tolist()
test_texts, test_labels = test_df["comments"].tolist(), test_df["Helpful"].tolist()
unlabeled_texts, unlabeled_labels = unlabled_df["comments"].tolist(),[0]*len(unlabled_df["comments"])

# Tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labeled_encodings = tokenizer(labeled_texts, truncation=True, padding=True)
labeled_dataset = Dataset(labeled_encodings, labeled_labels)
labeled_dataset.labels = F.one_hot(torch.tensor(labeled_dataset.labels),num_classes=2)

#We don't have dev and test split for now
dev_encodings = tokenizer(test_texts, truncation=True, padding=True)
dev_dataset = Dataset(dev_encodings, test_labels)
dev_dataset.labels = F.one_hot(torch.tensor(dev_dataset.labels),num_classes=2)

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = Dataset(test_encodings, test_labels)
test_dataset.labels = F.one_hot(torch.tensor(test_dataset.labels),num_classes=2)

unlabeled_encodings = tokenizer(unlabeled_texts, truncation=True, padding=True)
unlabeled_dataset = Dataset(unlabeled_encodings, unlabeled_labels)
unlabeled_dataset.labels = F.one_hot(torch.tensor(unlabeled_dataset.labels),num_classes=2)


# Initialize the Bert model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)
model = BERTClass(n_classes=2,dropout_rate=config.dropout_rate)
# Criterion & optimizer

# loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # or AdamW

#For baseline we can use alpha=0 and temp=1 as it translates to cross entropy loss
alpha = 0
temp = 1
if config.use_kd:
    alpha = config.kd_alpha
    temp = config.Temp
trainer = Trainer(config, model, optimizer, save_path, dev_dataset=test_dataset, test_dataset=test_dataset, alpha=alpha, temp=temp)

#Initial training
trainer.initial_train(labeled_dataset)

# load checkpoint
checkpoint_path = trainer.sup_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BERTClass(n_classes=2,dropout_rate=config.dropout_rate).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer
# print(labeled_dataset.labels)
# print(unlabeled_dataset.labels)
# eval supervised trained model
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)


# self-training
trainer.self_train(labeled_dataset, unlabeled_dataset)

# eval semi-supervised trained model
checkpoint_path = trainer.ssl_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BERTClass(n_classes=2,dropout_rate=config.dropout_rate).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)

