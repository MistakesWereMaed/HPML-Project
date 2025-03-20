import time
import torch
import pandas as pd

from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "metrics": metrics
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, weights_only=True)
    return checkpoint



def test(model, test_loader, loss_function, device="cuda"):
    model.to(device)
    model.eval()
    # Initialize return variables
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    # Testing loop
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            # Progress update
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    # Testing summary
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    return avg_loss, torch.cat(all_predictions), torch.cat(all_targets)

def validate(model, val, loss_function, device="cuda"):
    model.eval()
    val_loss = 0.0
    # Validation loop
    with torch.no_grad():
        for inputs, targets in val:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            val_loss += loss.item()
    # Validation summary
    avg_val_loss = val_loss / len(val)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train(model, loss_function, optimizer, train, val, num_epochs, path_model, patience=10, start_epoch=0):
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Load checkpoint
    try:
        checkpoint = load_checkpoint(f"{path_model}/Checkpoints/current.ckpt")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        metrics = checkpoint["metrics"]

        print(f"Resuming training from epoch {start_epoch} (Best val loss: {best_val_loss:.4f})")
    except FileNotFoundError:
        start_epoch = 0
        best_val_loss = float("inf")
        metrics = {"train_loss": [], "val_loss": [], "epoch_time": []}

        print("No checkpoint found, starting training from scratch.")
    # Gradient scaler for mixed precision
    scaler = torch.GradScaler(device)
    patience_counter = 0
    # Epoch training loop
    for epoch in range(start_epoch, num_epochs):
        # Epoch initialization
        torch.cuda.empty_cache()
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        start_time = time.time()
        # Inner training loop
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            with torch.autocast(device):
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Progress update
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        # Epoch summary
        end_time = time.time()
        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")
        # Validation and metrics
        val_loss = validate(model, val, loss_function, device)
        metrics["epoch_time"].append(end_time - start_time)
        metrics["train_loss"].append(avg_loss)
        metrics["val_loss"].append(val_loss)
        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            print(f"Updating best checkpoint...")
            #save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, f"{path_model}/Checkpoints/best.ckpt")
        else:
            patience_counter += 1
        #save_checkpoint(model, optimizer, epoch, val_loss, metrics, f"{path_model}/Checkpoints/current.ckpt")
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_val_loss:.4f}")
            break
    # Save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(f"{path_model}/metrics.csv", index=False)
    print("Training complete!")
    
    return metrics
