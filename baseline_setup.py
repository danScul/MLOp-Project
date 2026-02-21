#imports model architecture
from model import CNN

#imports dataset and seed setup
from data import get_data

#imports training and evaluation logic
from train_eval import train_and_evaluate

import csv
import os


#loads CIFAR-10 dataset and standardizes seed for consistency
trainloader, testloader = get_data(seed=1)


#creates instance of CNN model
model = CNN()


#runs full training workload and evaluation workload separately
train_runtime, eval_runtime, accuracy = train_and_evaluate(
    model,
    trainloader,
    testloader,
    epochs=5
)


#creates results folder if it does not already exist
os.makedirs("results", exist_ok=True)


#saves experiment results to CSV file for documentation
with open("results/baseline.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "train_runtime",
        "eval_runtime",
        "accuracy"
    ])
    writer.writerow([
        train_runtime,
        eval_runtime,
        accuracy
    ])


#prints final results to console
print("===== BASELINE COMPLETE =====")
print("Train runtime:", train_runtime)
print("Eval runtime:", eval_runtime)
print("Accuracy:", accuracy)