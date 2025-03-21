import torch

from tqdm import tqdm

# Testing Function
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