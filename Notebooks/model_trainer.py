import time
import torch

from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "metrics": metrics
    }
    torch.save(checkpoint, filename)
    print(f"Model checkpoint saved at epoch {epoch+1} with val_loss: {best_val_loss:.4f}")

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]
    metrics = checkpoint["metrics"]

    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch, loss, metrics




def validate(model, val, loss_function, device="cuda"):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(val, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train(model, loss_function, optimizer, train, val, num_epochs, path_weights_best, path_weights_last, start_epoch=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        start_epoch, best_val_loss, metrics = load_checkpoint(model, optimizer, path_weights_last)
        print(f"Resuming training from epoch {start_epoch} (Best val loss: {best_val_loss:.4f})")
    except FileNotFoundError:
        start_epoch = 0
        best_val_loss = float("inf")
        metrics = {"train_loss": [], "val_loss": [], "epoch_time": []}
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        start_time = time.time()
        
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = loss_function(predictions, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        end_time = time.time()
        metrics["epoch_time"].append(end_time - start_time)
        avg_loss = total_loss / len(train)
        metrics["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

        val_loss = validate(model, val, loss_function, device)
        metrics["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, path_weights_best)

        save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, path_weights_last)

    print("Training complete!")
    return metrics