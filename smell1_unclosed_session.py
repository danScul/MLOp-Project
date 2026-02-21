if __name__ == "__main__":

    #imports model architecture
    from model import CNN

    #imports dataset and seed setup
    from data import get_data

    #imports training and evaluation logic
    from train_eval import train_and_evaluate

    #emissions tracking library
    from codecarbon import EmissionsTracker

    import csv
    import os
    import psutil


    #loads CIFAR-10 dataset and standardizes seed for consistency
    trainloader, testloader = get_data(seed=1)


    #creates instance of CNN model
    model = CNN()


    #creates results folder if it does not already exist
    os.makedirs("results", exist_ok=True)


    #creates emissions tracker and starts tracking energy + CO2
    tracker = EmissionsTracker(output_dir="results")
    tracker.start()


    #runs full training workload and evaluation workload separately
    train_runtime, eval_runtime, accuracy = train_and_evaluate(
        model,
        trainloader,
        testloader,
        epochs=5
    )


    #stores model reference without deleting it (simulates unclosed session)
    stored_models = []
    stored_models.append(model)


    #stops emissions tracker and returns total CO2 (kg)
    emissions = tracker.stop()


    #creates a process object for the current program to measure RAM usage
    currentProcess = psutil.Process(os.getpid())

    #rss returns ram used by program, converted to megabytes
    peak_memory = currentProcess.memory_info().rss / (1024 ** 2)


    #saves experiment results to CSV file for documentation
    results_file = "results/smell1_unclosed_session.csv"
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
    print("Unclosed Session Leak Complete")
    print("Train runtime:", train_runtime)
    print("Eval runtime:", eval_runtime)
    print("Accuracy:", accuracy)
    print("CO2 emissions (kg):", emissions)
    print("Peak memory (MB):", peak_memory)