#!/usr/bin/env python3
"""Simple example demonstrating TopK SAE training on synthetic data.

This example shows how to use the TopK SAE without requiring large language models
or extensive setup. It's useful for understanding the basic functionality.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topk_sae.models.topk_sae import TopKSAE
from topk_sae.data.data_loader import ActivationDataset
from topk_sae.training.trainer import SAETrainer


def generate_synthetic_data(
    num_samples: int = 1000,
    input_dim: int = 128,
    num_features: int = 32,
    sparsity: float = 0.1,
) -> torch.Tensor:
    """Generate synthetic sparse activation data.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Dimension of input activations
        num_features: Number of underlying sparse features
        sparsity: Fraction of features active per sample
        
    Returns:
        Synthetic activation tensor
    """
    print(f"Generating {num_samples} synthetic samples...")
    print(f"Input dim: {input_dim}, Features: {num_features}, Sparsity: {sparsity}")
    
    # Create feature dictionary (features are random vectors)
    feature_dict = torch.randn(num_features, input_dim)
    feature_dict = F.normalize(feature_dict, dim=1)  # Normalize features
    
    activations = []
    for _ in range(num_samples):
        # Randomly select which features are active
        num_active = int(sparsity * num_features)
        active_features = torch.randperm(num_features)[:num_active]
        
        # Generate random feature strengths
        strengths = torch.randn(num_active).abs() + 0.5  # Positive strengths
        
        # Create activation as weighted sum of features
        activation = torch.zeros(input_dim)
        for i, feature_idx in enumerate(active_features):
            activation += strengths[i] * feature_dict[feature_idx]
            
        # Add small amount of noise
        activation += 0.1 * torch.randn(input_dim)
        
        activations.append(activation)
        
    return torch.stack(activations)


def main() -> None:
    """Run the synthetic data example."""
    print("TopK SAE Synthetic Data Example")
    print("=" * 40)
    
    # Generate synthetic data
    activations = generate_synthetic_data(
        num_samples=2000,
        input_dim=64,
        num_features=16,
        sparsity=0.2  # 20% of features active
    )
    
    print(f"Generated data shape: {activations.shape}")
    print(f"Data mean: {activations.mean():.3f}")
    print(f"Data std: {activations.std():.3f}")
    
    # Create dataset and data loaders
    dataset = ActivationDataset(activations)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create TopK SAE model
    input_dim = activations.shape[-1]
    hidden_dim = input_dim * 4  # 4x overcomplete
    k = hidden_dim // 8  # ~12.5% sparsity
    
    model = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k=k,
        tied_weights=True,
        normalize_decoder=True,
    )
    
    print(f"\nModel Configuration:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Top-k: {model.k}")
    print(f"  Sparsity: {model.k / model.hidden_dim:.1%}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = SAETrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device="cpu",  # Use CPU for this example
        log_dir="logs/example",
        checkpoint_dir="checkpoints/example",
        use_wandb=False,  # Disable wandb for example
        l1_lambda=1e-4,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {trainer.device}")
    print(f"  L1 lambda: {trainer.l1_lambda}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Train model
    print(f"\nStarting training...")
    trainer.train(
        num_epochs=10,
        eval_every=50,
        save_every=200,
        log_every=25,
    )
    
    # Evaluate final model
    print(f"\nFinal Evaluation:")
    val_metrics = trainer.evaluate(val_loader)
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
        
    # Feature analysis
    print(f"\nFeature Analysis:")
    analysis = trainer.analyze_features(val_loader)
    print(f"  Dead features: {analysis['num_dead_features']}")
    print(f"  Sparsity: {analysis['sparsity']:.3f}")
    print(f"  MSE: {analysis['overall_mse']:.4f}")
    print(f"  Cosine similarity: {analysis['cosine_similarity']:.4f}")
    print(f"  Explained variance: {analysis['explained_variance']:.4f}")
    
    # Test reconstruction on a few samples
    print(f"\nReconstruction Examples:")
    model.eval()
    with torch.no_grad():
        test_batch = activations[:5]  # First 5 samples
        reconstruction, sparse_hidden, loss_dict = model(test_batch)
        
        for i in range(len(test_batch)):
            original = test_batch[i]
            recon = reconstruction[i]
            sparse = sparse_hidden[i]
            
            mse = F.mse_loss(recon, original).item()
            cosine_sim = F.cosine_similarity(original, recon, dim=0).item()
            num_active = (sparse > 0).sum().item()
            
            print(f"  Sample {i+1}:")
            print(f"    MSE: {mse:.4f}")
            print(f"    Cosine sim: {cosine_sim:.4f}")
            print(f"    Active features: {num_active}/{model.hidden_dim}")
    
    print(f"\nExample completed successfully!")
    print(f"Checkpoints saved to: checkpoints/example/")
    print(f"Logs saved to: logs/example/")


if __name__ == "__main__":
    main() 