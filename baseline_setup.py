if __name__ == "__main__":
    #imports model architecture
    from codecarbon import EmissionsTracker

    from model import CNN

    #imports dataset and seed setup
    from data import get_data

    #imports training and evaluation logic
    from train_eval import train_and_evaluate

    import csv
    import os
    import psutil #for peak memory tracking

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


    #Emissions Tracking - CO2 and Energy consumption
    tracker = EmissionsTracker() #Creates tracker writes to a file named emissions saved to the current folder
    tracker.start()
    train_runtime, eval_runtime, accuracy = train_and_evaluate(model, trainloader, testloader) #run train and eval, save each of the outputs to the respective variables
    
    tracker.stop() 

    with open("emissions.csv", "r") as f:
        reader = csv.DictReader(f)
        last_row = list(reader)[-1]  # get the most recent run

    energy_kwh = float(last_row["energy_consumed"])
    co2_kg = float(last_row["emissions"])

    currentProcess = psutil.Process(os.getpid()) #Creates a process object for the current program uses the Process ID to determine how much RAM is allocated to our program
    peak_memory = currentProcess.memory_info().rss / (1024 ** 2) #returns memory usage information in form of megabytes for condensing purposes (rss returns ram used)

    with open("results.csv", "a", newline="") as f: #creates a new csv file and configures settings to properly write results to it
        writer = csv.writer(f) #creates writer
        writer.writerow([
            train_runtime,
            eval_runtime,
            accuracy,
            energy_kwh,
            co2_kg,
            peak_memory
        ])

    #prints final results to console
    print("===== BASELINE COMPLETE =====")
    print("Train runtime:", train_runtime)
    print("Eval runtime:", eval_runtime)
    print("Accuracy:", accuracy)