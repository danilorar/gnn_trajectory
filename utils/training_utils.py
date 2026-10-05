import numpy as np 
import torch  
import matplotlib.pyplot as plt  


# For loss function
def ade_loss(predictions, targets): 
    """Compute the Average Displacement Error (ADE) between predictions and targets."""
    errors = torch.norm(predictions - targets, dim=2)
    return errors.mean()  # Average over all timesteps and samples


# Training and validation loop
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs, checkpoint_path, device=None): 
    """Train and validate a model for multiple epochs. Saves the parameters of the best model based on validation loss."""
    
    if device is None:
        device = next(model.parameters()).device
        
    # Init 
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize best validation loss to infinity 
    
    # loop 
    for epoch in range(num_epochs):
        
        # ==================
        # ==== TRAINING ====
        # ==================
         
        model.train()  # set training mode
        
        train_total_loss = 0.0
        train_examples = 0
        
        for batch in train_loader:
            
            batch = batch.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            
            pred = model(batch)  # Forward pass
            
            target = batch.y.view(batch.num_graphs, 12, 2)  # Reshape target to match prediction shape
            
            loss = criterion(pred, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            
            train_total_loss += (loss.item() * batch.num_graphs)  # Accumulate total loss
            train_examples += batch.num_graphs  # Accumulate number of examples
            
        # Compute average training loss for the epoch
        train_avg_loss = (train_total_loss / train_examples)
        train_losses.append(train_avg_loss)
        
        
        # ====================
        # ==== VALIDATION ====
        # ====================

        model.eval()  # set evaluation mode
        
        val_total_loss = 0.0
        val_examples = 0
        
        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_loader: 
                
                batch = batch.to(device)
                
                pred = model(batch)  # Forward pass
                
                target = batch.y.view(batch.num_graphs, 12, 2)  # Reshape target
                
                loss = criterion(pred, target)  # Compute loss
                
                val_total_loss += (loss.item() * batch.num_graphs)  # Accumulate total loss
                val_examples += batch.num_graphs  # Accumulate number of examples
                
        # Compute average validation loss for the epoch
        val_avg_loss = (val_total_loss / val_examples)
        val_losses.append(val_avg_loss)
        
        # Print 
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train ADE: {train_avg_loss:.4f} m | "
            f"Val ADE: {val_avg_loss:.4f} m"
        )
        
        # SAVE BEST CHECKPOINT
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(
                model.state_dict(),
                checkpoint_path
            )

    return train_losses, val_losses

# ========================================================

def evaluate_model(model, loader, device=None):
    """Compute mean ADE and FDE for a dataset."""

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    ade_list = []
    fde_list = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)
            
            pred = model(batch) # Forward pass
            target = batch.y.view(batch.num_graphs, 12, 2) 

            errors = torch.norm(pred - target, dim=2)  # Euclidean distance for each timestep
            
            # Compute ADE and FDE for each sample
            ade_per_sample = errors.mean(dim=1)
            fde_per_sample = errors[:, -1]

            ade_list.extend(ade_per_sample.cpu().numpy())
            fde_list.extend(fde_per_sample.cpu().numpy())

    mean_ade = np.mean(ade_list)
    mean_fde = np.mean(fde_list)

    return mean_ade, mean_fde

# ========================================================

def plot_losses(train_losses,val_losses,title="Training and Validation Losses"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Train ADE")
    
    plt.plot(val_losses, label="Validation ADE" )

    plt.xlabel("Epoch")
    plt.ylabel("ADE (m)")
    plt.title(title)

    plt.legend()
    plt.grid()

    plt.show()