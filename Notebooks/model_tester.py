import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler

from models import PICPModel
from model_trainer import load_checkpoint, get_unused_port, setup, cleanup
from data_loader import load_dataset

PATH_TEST = "../Data/Processed/Test.nc"
PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

def test(rank, world_size, port, model, name, loss_function, optimizer, test_ds, batch_size, return_dict):
    setup(rank, world_size, port)
    
    model.to(rank)
    model = FSDP(model)
    load_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer)

    sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=sampler)
    
    # Initialize return variables
    total_loss = torch.tensor(0.0, device=rank)
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(test_loader, desc="Testing", leave=False) if rank == 0 else test_loader
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(rank), targets.to(rank)
            # Forward pass
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            # Accumulate loss tensor
            total_loss += loss.detach()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            if rank == 0:
                progress_bar.set_postfix(loss=total_loss.item())

    # Average loss across all ranks
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= dist.get_world_size()
    
    # Gather results on rank 0
    if rank == 0:
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss.item():.4f}")
        
        # Collect all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return_dict["loss"] = avg_loss.item()
        return_dict["predictions"] = all_predictions
        return_dict["targets"] = all_targets

    cleanup()
    torch.cuda.empty_cache()

def main(args):
    # Select model
    model_type = args["model"]
    match model_type:
        case "PINN":
            model_class = PICPModel
        #case "GNN": 
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type")
    
    # Load Data
    params = model_class.load_params()
    test_ds, image_size = load_dataset(path=PATH_TEST, input_days=params["input_days"], target_days=params["target_days"])
    
    # Initialize model
    model_kwargs = model_class.initialize_model(image_size, params)
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    
    # Setup DDP environment to load model weights
    port = get_unused_port()
    world_size = torch.cuda.device_count()

    return_dict = mp.Manager().dict()

    # Start the testing process
    if world_size > 1:
        mp.spawn(test, args=(world_size, port, model, name, loss_function, optimizer, test_ds, params["batch_size"], return_dict), nprocs=world_size, join=True)
    else:
        test(0, world_size, port, model, name, loss_function, optimizer, test_ds, params["batch_size"], return_dict)

    # Collect results
    loss = return_dict["loss"]
    predictions = return_dict["predictions"]
    targets = return_dict["targets"]
    
    # Save results
    results_path = f"{PATH_RESULTS}/{model_type}.npz"
    np.savez_compressed(
        results_path,
        loss=np.array(loss, dtype=np.float32),
        predictions=predictions.cpu().numpy(),
        targets=targets.cpu().numpy()
    )

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    args = parser.parse_args()
    
    main(vars(args))