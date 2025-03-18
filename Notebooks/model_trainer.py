import torch
from tqdm import tqdm

def save_checkpoint(model, optimizer, epoch, best_val_loss, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, filename)
    print(f"Model checkpoint saved at epoch {epoch+1} with val_loss: {best_val_loss:.4f}")

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]

    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch, loss




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
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, path_weights_last)
        print(f"Resuming training from epoch {start_epoch} (Best val loss: {best_val_loss:.4f})")
    except FileNotFoundError:
        start_epoch = 0
        best_val_loss = float("inf")
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0

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

        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

        val_loss = validate(model, val, loss_function, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, path_weights_best)

        save_checkpoint(model, optimizer, epoch, best_val_loss, path_weights_last)

    print("Training complete!")