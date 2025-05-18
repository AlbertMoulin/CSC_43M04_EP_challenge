import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import hydra

def visualize_branch_weights(model, checkpoint_path=None):
    """
    Load a model and visualize the learned weights for each branch.
    
    Args:
        model: The CombinedModel instance or class to examine
        checkpoint_path: Optional path to a saved checkpoint
    """
    # Load from checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(r"checkpoints/EnhancedCombinedModel_2025-05-18_17-48-47_best.pt", map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
    # Get the raw weights
    raw_weights = torch.stack([
        model.weight_image,
        model.weight_text,
        model.weight_metadata
    ]).detach().cpu().numpy()
    
    # Apply softmax to get the normalized weights
    softmax_weights = torch.softmax(
        torch.tensor(raw_weights),
        dim=0
    ).numpy()
    
    # Create a dataframe for better display
    weights_df = pd.DataFrame({
        'Branch': ['Image', 'Text', 'Metadata'],
        'Raw Weight': raw_weights,
        'Normalized Weight': softmax_weights
    })
    
    print("Branch Importance Weights:")
    print(weights_df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        weights_df['Branch'], 
        weights_df['Normalized Weight'], 
        color=['#3498db', '#2ecc71', '#e74c3c']
    )
    
    # Add weight values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    plt.title('Importance of Each Branch in the Combined Model', fontsize=16)
    plt.ylabel('Normalized Weight', fontsize=14)
    plt.ylim(0, max(softmax_weights) * 1.2)  # Add some space above bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('branch_weights.png')
    plt.show()
    
    return weights_df

# For tracking weights throughout training
def track_weights_during_training(model, train_loader, epochs=5):
    """
    Track how weights evolve during training.
    This is a simplified version that just records weights after each epoch.
    
    Args:
        model: The CombinedModel instance
        train_loader: DataLoader for training data
        epochs: Number of epochs to track
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    # Store weights over time
    weight_history = {
        'epoch': [],
        'image': [],
        'text': [],
        'metadata': []
    }
    
    # Initial weights
    weights = torch.softmax(
        torch.stack([model.weight_image, model.weight_text, model.weight_metadata]),
        dim=0
    ).detach().cpu().numpy()
    
    weight_history['epoch'].append(0)
    weight_history['image'].append(weights[0])
    weight_history['text'].append(weights[1])
    weight_history['metadata'].append(weights[2])
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move tensors to device
            batch["image"] = batch["image"].to(device)
            if "target" in batch:
                batch["target"] = batch["target"].to(device).squeeze()
            
            # Forward pass
            preds = model(batch).squeeze()
            
            # If this is just for visualization, we can skip the backward pass
            if "target" in batch:
                loss = loss_fn(preds, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Record weights after each epoch
        weights = torch.softmax(
            torch.stack([model.weight_image, model.weight_text, model.weight_metadata]),
            dim=0
        ).detach().cpu().numpy()
        
        weight_history['epoch'].append(epoch+1)
        weight_history['image'].append(weights[0])
        weight_history['text'].append(weights[1])
        weight_history['metadata'].append(weights[2])
    
    # Plot weight evolution
    plt.figure(figsize=(12, 7))
    plt.plot(weight_history['epoch'], weight_history['image'], 'o-', label='Image', linewidth=2)
    plt.plot(weight_history['epoch'], weight_history['text'], 's-', label='Text', linewidth=2)
    plt.plot(weight_history['epoch'], weight_history['metadata'], '^-', label='Metadata', linewidth=2)
    
    plt.title('Evolution of Branch Weights During Training', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Normalized Weight', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('weight_evolution.png')
    plt.show()
    
    return pd.DataFrame(weight_history)

# Example usage after training a model:
if __name__ == "__main__":
    # Option 1: Use Hydra to load a model from config
    @hydra.main(config_path="configs", config_name="train")
    def analyze_model_weights(cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        
        # Update model config with number of channels from dataset
        train_dataset = datamodule.train_dataloader().dataset
        if hasattr(train_dataset, 'dataset'):  # Handle case of Subset wrapping
            num_channels = train_dataset.dataset.get_num_channels()
        else:
            num_channels = train_dataset.get_num_channels()
        
        if 'num_channels' in cfg.model.instance:
            cfg.model.instance.num_channels = num_channels
        
        # Initialize model
        model = hydra.utils.instantiate(cfg.model.instance).to(device)
        
        # Load checkpoint if specified
        if cfg.checkpoint_path:
            visualize_branch_weights(model, cfg.checkpoint_path)
        else:
            visualize_branch_weights(model)
            
        # Optionally track weights during a few epochs of training
        # Uncomment the following to see how weights evolve
        # train_loader = datamodule.train_dataloader()
        # track_weights_during_training(model, train_loader, epochs=3)
        
    analyze_model_weights()
    
    # Option 2: Directly load a specific model and checkpoint
    """
    from models.combined_model import EnhancedCombinedModel
    
    model = EnhancedCombinedModel(
        image_mlp_layers=[1024, 512, 256, 1],
        text_model_name='facebook/bart-base',
        text_mlp_layers=[1024, 512, 256, 1],
        metadata_mlp_layers=[1024, 512, 256, 1],
        num_channels=1000  # Will be updated with actual value
    )
    
    checkpoint_path = "checkpoints/EnhancedCombinedModel_2025-05-16_best.pt"
    visualize_branch_weights(model, checkpoint_path)
    """