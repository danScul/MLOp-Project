import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_and_evaluate(model, trainloader, testloader, epochs=5):

    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ================= TRAINING =================
    train_start = time.time()

    for epoch in range(epochs):
        model.train() #sets model to training mode

        for inputs, labels in trainloader:
            optimizer.zero_grad() #clears previous gradients
            outputs = model(inputs) #forward pass
            loss = criterion(outputs, labels)
            loss.backward() #backpropagation
            optimizer.step() #updates weights

    train_runtime = time.time() - train_start

    # ================= EVALUATION =================
    eval_start = time.time()

    model.eval() #sets model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad(): #turns off gradient tracking
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    eval_runtime = time.time() - eval_start

    return train_runtime, eval_runtime, accuracy