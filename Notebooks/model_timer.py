import time
import csv
import argparse
import torch
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model_trainer import setup, cleanup, get_unused_port, train_epoch
from models import PICPModel
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"

GPU_CONFIGS = [1, 2, 4]

def train_timing(rank, world_size, port, model, loss_function, optimizer, train_ds, batch_size, epochs, train_times):
    # Initialize DDP
    setup(rank, world_size, port)
    model.to(f"cuda:{rank}")
    model = DDP(model, device_ids=[rank])
    # Initialize DataLoader
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    scaler = torch.GradScaler()
    start_time = time.perf_counter()
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_epoch(rank, model, train_loader, loss_function, optimizer, scaler)

    end_time = time.perf_counter()
    # Save results
    train_time = end_time - start_time
    train_times[rank] = train_time / epochs
    # Cleanup
    del model, loss_function, optimizer
    torch.cuda.empty_cache()
    cleanup()

def main(args):
    model_type = args["model"]
    epochs = args["epochs"]
    trials = args["trials"]
    # Select model class
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")
    # Load Data
    params = model_class.load_params()
    train_ds, image_size = load_dataset(path=PATH_TRAIN, input_days=params["input_days"], target_days=params["target_days"])
    # Train with different GPU configurations
    results = []
    for world_size in GPU_CONFIGS:
        if world_size > torch.cuda.device_count():
            print(f"Skipping {world_size} GPUs (only {torch.cuda.device_count()} available)")
            continue
        print(f"\nTraining with {world_size} GPU(s)...")
        # Average multiple trials
        total_time = 0
        for trial in range(trials):
            # Initialize model
            model_kwargs = model_class.initialize_model(image_size, params)
            model = model_kwargs["model"]
            loss_function = model_kwargs["loss_function"]
            optimizer = model_kwargs["optimizer"]
            # Shared dictionary for results
            with mp.Manager() as manager:
                train_times = manager.dict()
                # Launch training on multiple GPUs
                port = get_unused_port()
                if world_size > 1:
                    mp.spawn(train_timing, args=(world_size, port, model, loss_function, optimizer, train_ds, params["batch_size"], epochs, train_times), nprocs=world_size, join=True)
                else:
                    train_timing(0, world_size, port, model, loss_function, optimizer, train_ds, params["batch_size"], epochs, train_times)
                # Aggregate training times from the shared dictionary
                total_time += sum(train_times.values())
        # Compute average time for this GPU configuration
        avg_time = total_time / trials / world_size
        results.append([world_size, avg_time])
        print(f"Average time per epoch for {world_size} GPU(s): {avg_time:.2f} seconds")
    # Save results to CSV
    with open(f"{PATH_TIMINGS}/{model_type}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "train_time"])
        writer.writerows(results)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    main(vars(args))