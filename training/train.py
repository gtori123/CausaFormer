import torch
import torch.nn as nn
import torch.optim as optim
from models.causaformer import CausaFormer

def generate_dummy_data(batch_size, seq_len, input_dim):
    """
    Generate dummy input and target data for training.
    Args:
        batch_size (int): Number of samples in a batch.
        seq_len (int): Length of the sequence.
        input_dim (int): Dimensionality of each input vector.
    Returns:
        input_data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, input_dim).
        target_data (torch.Tensor): Target data tensor of shape (batch_size, seq_len, input_dim).
    """
    input_data = torch.randn(batch_size, seq_len, input_dim)
    target_data = torch.randn(batch_size, seq_len, input_dim)
    return input_data, target_data

def train_model(model, criterion, optimizer, num_epochs=10, batch_size=2, seq_len=10, input_dim=512):
    """
    Train the CausaFormer model with dummy data.
    Args:
        model (torch.nn.Module): CausaFormer model instance.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        batch_size (int): Number of samples in a batch.
        seq_len (int): Length of the sequence.
        input_dim (int): Dimensionality of each input vector.
    """
    for epoch in range(num_epochs):
        # Generate dummy data
        input_data, target_data = generate_dummy_data(batch_size, seq_len, input_dim)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)

        # Compute loss
        loss = criterion(output, target_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def main():
    # Model configuration
    input_dim = 512
    num_layers = 4
    n_heads = 8
    dropout = 0.1
    batch_size = 2
    seq_len = 10
    num_epochs = 10
    learning_rate = 1e-3

    # Initialize the model
    model = CausaFormer(input_dim=input_dim, num_layers=num_layers, n_heads=n_heads, dropout=dropout)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train_model(model, criterion, optimizer, num_epochs=num_epochs, batch_size=batch_size, seq_len=seq_len, input_dim=input_dim)
    print("Training complete.")

if __name__ == "__main__":
    main()
