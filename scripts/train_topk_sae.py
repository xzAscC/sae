#!/usr/bin/env python3
"""Main training script for TopK SAE on Gemma2-2b using nnsight.

This script demonstrates how to train a TopK Sparse Autoencoder on activations
from the Gemma2-2b language model using the nnsight library.

Usage:
    python scripts/train_topk_sae.py --config configs/gemma2_2b.yaml
    python scripts/train_topk_sae.py --quick-test  # For quick testing
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topk_sae.models.topk_sae import TopKSAE
from topk_sae.data.data_loader import ActivationDataLoader, ActivationDataset
from topk_sae.training.trainer import SAETrainer
from topk_sae.config.config import (
    FullConfig,
    get_gemma2_2b_config,
    get_small_test_config,
)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train TopK SAE on Gemma2-2b activations"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with small configuration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified"
    )
    
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, don't train"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-activations",
        type=str,
        help="Path to save collected activations"
    )
    
    parser.add_argument(
        "--load-activations",
        type=str,
        help="Path to load pre-collected activations"
    )
    
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> FullConfig:
    """Load configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object
    """
    if args.quick_test:
        logging.info("Using quick test configuration")
        config = get_small_test_config()
    elif args.config:
        logging.info(f"Loading configuration from {args.config}")
        config = FullConfig.load_yaml(args.config)
    else:
        logging.info("Using default Gemma2-2b configuration")
        config = get_gemma2_2b_config()
        
    # Override device if specified
    if args.device:
        logging.info(f"Overriding device to {args.device}")
        
    return config


def prepare_data(
    config: FullConfig, 
    args: argparse.Namespace
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Prepare training and validation data.
    
    Args:
        config: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if args.load_activations:
        # Load pre-collected activations
        logging.info(f"Loading activations from {args.load_activations}")
        
        activations = torch.load(args.load_activations, map_location="cpu")
        dataset = ActivationDataset(activations)
        
        logging.info(f"Loaded {len(dataset)} activation samples")
        
    else:
        # Collect activations from language model
        logging.info("Collecting activations from language model...")
        
        # Initialize data loader
        data_loader = ActivationDataLoader(
            model_name=config.data.model_name,
            layer_name=config.data.layer_name,
            device=args.device,
            cache_dir=config.data.cache_dir,
        )
        
        # Create dataset
        dataset = data_loader.create_dataset_from_text(
            dataset_name=config.data.dataset_name,
            split=config.data.dataset_split,
            num_samples=config.data.num_samples,
            streaming=config.data.streaming,
            max_length=config.data.max_length,
            batch_size=config.data.batch_size,
        )
        
        # Save activations if requested
        if args.save_activations and not config.data.streaming:
            data_loader.save_activations(dataset.activations, args.save_activations)
            
        logging.info(f"Collected {len(dataset)} activation samples")
        
    # Split into train/validation
    if config.data.streaming:
        # For streaming, we can't easily split, so use the whole dataset for training
        train_loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,  # Streaming datasets can't be shuffled
        )
        val_loader = None
        logging.info("Using streaming dataset for training (no validation split)")
        
    else:
        # Split dataset
        train_size = int(config.training.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        
    return train_loader, val_loader


def create_model(config: FullConfig, activation_dim: Optional[int] = None) -> TopKSAE:
    """Create TopK SAE model.
    
    Args:
        config: Configuration object
        activation_dim: Dimension of activations (auto-detected if None)
        
    Returns:
        TopK SAE model
    """
    if activation_dim is None:
        # Try to get from config, otherwise use a default
        activation_dim = config.model.input_dim
        
    logging.info(f"Creating TopK SAE model:")
    logging.info(f"  Input dim: {activation_dim}")
    logging.info(f"  Hidden dim: {config.model.hidden_dim}")
    logging.info(f"  Top-k: {config.model.k}")
    logging.info(f"  Tied weights: {config.model.tied_weights}")
    logging.info(f"  Normalize decoder: {config.model.normalize_decoder}")
    
    # Create model
    model = TopKSAE(
        input_dim=activation_dim,
        hidden_dim=config.model.hidden_dim,
        k=config.model.k,
        tied_weights=config.model.tied_weights,
        normalize_decoder=config.model.normalize_decoder,
        bias=config.model.bias,
        dtype=getattr(torch, config.model.dtype),
    )
    
    return model


def create_optimizer(
    model: TopKSAE, config: FullConfig
) -> torch.optim.Optimizer:
    """Create optimizer for training.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Optimizer instance
    """
    if config.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
        
    logging.info(f"Created {config.training.optimizer} optimizer with lr={config.training.learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: FullConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration object
        
    Returns:
        Scheduler instance or None
    """
    if config.training.scheduler is None:
        return None
        
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.scheduler_params.get("T_max", 1000),
            eta_min=config.training.scheduler_params.get("eta_min", 1e-6),
        )
    elif config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_params.get("step_size", 1000),
            gamma=config.training.scheduler_params.get("gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.training.scheduler}")
        
    logging.info(f"Created {config.training.scheduler} scheduler")
    return scheduler


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    logging.info("Starting TopK SAE training for Gemma2-2b")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA devices: {torch.cuda.device_count()}")
        
    # Load configuration
    config = load_configuration(args)
    
    # Save configuration for reproducibility
    os.makedirs("configs", exist_ok=True)
    config_save_path = "configs/current_config.yaml"
    config.save_yaml(config_save_path)
    logging.info(f"Saved current configuration to {config_save_path}")
    
    try:
        # Prepare data
        train_loader, val_loader = prepare_data(config, args)
        
        # Auto-detect activation dimension from data
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_batch = sample_batch[0]
        activation_dim = sample_batch.shape[-1]
        
        logging.info(f"Auto-detected activation dimension: {activation_dim}")
        
        # Update config with detected dimension
        config.model.input_dim = activation_dim
        
        # Create model
        model = create_model(config, activation_dim)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        # Create trainer
        trainer = SAETrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            log_dir=config.training.log_dir,
            checkpoint_dir=config.training.checkpoint_dir,
            use_wandb=config.training.use_wandb,
            wandb_project=config.training.wandb_project,
            l1_lambda=config.training.l1_lambda,
            grad_clip_norm=config.training.grad_clip_norm,
        )
        
        # Load checkpoint if specified
        if args.load_checkpoint:
            logging.info(f"Loading checkpoint from {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)
            
        # Run training or evaluation
        if args.eval_only:
            if val_loader is None:
                logging.error("No validation data available for evaluation")
                sys.exit(1)
                
            logging.info("Running evaluation only...")
            metrics = trainer.evaluate(val_loader)
            
            logging.info("Evaluation results:")
            for key, value in metrics.items():
                logging.info(f"  {key}: {value:.4f}")
                
            # Feature analysis
            logging.info("Running feature analysis...")
            analysis = trainer.analyze_features(val_loader, num_batches=10)
            
            logging.info("Feature analysis results:")
            logging.info(f"  Dead features: {analysis['num_dead_features']}")
            logging.info(f"  Sparsity: {analysis['sparsity']:.3f}")
            logging.info(f"  MSE: {analysis['overall_mse']:.4f}")
            logging.info(f"  Cosine similarity: {analysis['cosine_similarity']:.4f}")
            logging.info(f"  Explained variance: {analysis['explained_variance']:.4f}")
            
        else:
            # Train model
            logging.info("Starting training...")
            trainer.train(
                num_epochs=config.training.num_epochs,
                eval_every=config.training.eval_every,
                save_every=config.training.save_every,
                log_every=config.training.log_every,
                max_steps=config.training.max_steps,
            )
            
            logging.info("Training completed successfully!")
            
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
        
    logging.info("Script completed successfully!")


if __name__ == "__main__":
    main() 