if __name__ == "__main__":

    #imports model architecture
    from model import CNN

    #imports dataset and seed setup
    from data import get_data

    #emissions tracking library
    from codecarbon import EmissionsTracker

    import torch
    import csv
    import os
    import psutil
    import time


    #loads CIFAR-10 dataset and standardizes seed for consistency
    trainloader, testloader = get_data(seed=1)


    #creates instance of CNN model
    model = CNN()


    #creates results folder if it does not already exist
    os.makedirs("results", exist_ok=True)


    #creates emissions tracker and starts tracking energy + CO2
    tracker = EmissionsTracker(output_dir="results")
    tracker.start()


    #creates list to store image batches (simulates image buffer accumulation)
    stored_images = []


    #loss function
    criterion = torch.nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5


    #training workload
    train_start = time.time()

    for epoch in range(epochs):

        model.train()

        for inputs, labels in trainloader:

            #stores image batch without clearing (resource leak)
            stored_images.append(inputs.clone())

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_runtime = time.time() - train_start


    #evaluation workload
    eval_start = time.time()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    eval_runtime = time.time() - eval_start


    #stops emissions tracker and returns total CO2 (kg)
    emissions = tracker.stop()


    #creates a process object for the current program to measure RAM usage
    currentProcess = psutil.Process(os.getpid())

    #rss returns ram used by program, converted to megabytes
    peak_memory = currentProcess.memory_info().rss / (1024 ** 2)


    #saves experiment results to CSV file for documentation
    results_file = "results/smell2_image_buffer.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)

        #writes header only if file does not already exist
        if not file_exists:
            writer.writerow([
                "train_runtime",
                "eval_runtime",
                "accuracy",
                "co2_emissions_kg",
                "peak_memory_MB"
            ])

        writer.writerow([
            train_runtime,
            eval_runtime,
            accuracy,
            emissions,
            peak_memory
        ])


    #prints final results to console
    print("Image Buffer Accumulation Complete")
    print("Train runtime:", train_runtime)
    print("Eval runtime:", eval_runtime)
    print("Accuracy:", accuracy)
    print("CO2 emissions (kg):", emissions)
    print("Peak memory (MB):", peak_memory)