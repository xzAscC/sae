#!/usr/bin/env python3
"""Training script for TopK SAE on PaCE-1M concept dataset.

This script trains TopK Sparse Autoencoders on the PaCE-1M concept dataset,
which contains concept representations from the Parsimonious Concept Engineering
project.

Usage:
    python scripts/train_pace_sae.py --config configs/pace_1m.yaml
    python scripts/train_pace_sae.py --download-dataset  # Download dataset first
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topk_sae.models.topk_sae import TopKSAE
from topk_sae.training.trainer import SAETrainer
from topk_sae.data.concept_data_loader import PaCEConceptDataLoader, create_pace_config
from topk_sae.config.config import FullConfig


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
            logging.FileHandler("pace_training.log")
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train TopK SAE on PaCE-1M concept dataset"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download PaCE-1M dataset before training"
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
        "--max-concepts",
        type=int,
        help="Maximum number of concepts to use for training"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with small number of concepts"
    )
    
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> dict:
    """Load configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    if args.config:
        logging.info(f"Loading configuration from {args.config}")
        # Load from YAML file
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logging.info("Using default PaCE-1M configuration")
        config = create_pace_config()
        
    # Override with command line arguments
    if args.max_concepts:
        config["data"]["max_concepts"] = args.max_concepts
        
    if args.quick_test:
        config["data"]["max_concepts"] = 100  # Small test
        config["training"]["num_epochs"] = 2
        config["training"]["batch_size"] = 32
        config["training"]["eval_every"] = 50
        config["training"]["save_every"] = 100
        config["training"]["use_wandb"] = False
        
    return config


def download_dataset_if_needed(args: argparse.Namespace) -> None:
    """Download PaCE-1M dataset if requested.
    
    Args:
        args: Parsed command line arguments
    """
    if args.download_dataset:
        logging.info("Downloading PaCE-1M dataset...")
        
        loader = PaCEConceptDataLoader(
            concept_dir="./concept",
            concept_index_path="./concept_index.txt"
        )
        
        loader.download_and_extract_dataset("./")
        
        logging.info("Dataset download completed!")
    else:
        # Check if dataset exists
        concept_dir = Path("./concept")
        concept_index = Path("./concept_index.txt")
        
        if not concept_dir.exists() or not concept_index.exists():
            logging.warning("PaCE-1M dataset not found!")
            logging.warning("Run with --download-dataset to download the dataset")
            logging.warning("Or manually download from: https://github.com/peterljq/Parsimonious-Concept-Engineering")


def prepare_data(config: dict, args: argparse.Namespace) -> tuple[DataLoader, Optional[DataLoader]]:
    """Prepare training and validation data.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logging.info("Preparing PaCE-1M concept data...")
    
    # Create data loader
    data_loader = PaCEConceptDataLoader(
        concept_dir=config["data"]["concept_dir"],
        concept_index_path=config["data"]["concept_index_path"],
    )
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        max_concepts=config["data"]["max_concepts"],
        batch_size=config["training"]["batch_size"],
        train_val_split=config["data"]["train_val_split"],
        normalize=config["data"]["normalize"],
    )
    
    # Analyze dataset
    sample_batch = next(iter(train_loader))
    representation_dim = sample_batch.shape[-1]
    
    logging.info(f"Auto-detected representation dimension: {representation_dim}")
    logging.info(f"Train batches: {len(train_loader)}")
    logging.info(f"Val batches: {len(val_loader)}")
    
    # Update config with detected dimension
    config["model"]["input_dim"] = representation_dim
    
    return train_loader, val_loader


def create_model(config: dict) -> TopKSAE:
    """Create TopK SAE model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TopK SAE model
    """
    model_config = config["model"]
    
    logging.info(f"Creating TopK SAE model:")
    logging.info(f"  Input dim: {model_config['input_dim']}")
    logging.info(f"  Hidden dim: {model_config['hidden_dim']}")
    logging.info(f"  Top-k: {model_config['k']}")
    logging.info(f"  Sparsity: {model_config['k'] / model_config['hidden_dim']:.1%}")
    
    model = TopKSAE(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        k=model_config["k"],
        tied_weights=model_config["tied_weights"],
        normalize_decoder=model_config["normalize_decoder"],
        bias=model_config["bias"],
        dtype=getattr(torch, model_config["dtype"]),
    )
    
    return model


def create_optimizer(model: TopKSAE, config: dict) -> torch.optim.Optimizer:
    """Create optimizer for training.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    training_config = config["training"]
    
    if training_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
    elif training_config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
        
    logging.info(f"Created {training_config['optimizer']} optimizer with lr={training_config['learning_rate']}")
    return optimizer


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    logging.info("Starting TopK SAE training on PaCE-1M concept dataset")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA devices: {torch.cuda.device_count()}")
        
    # Download dataset if needed
    download_dataset_if_needed(args)
    
    # Load configuration
    config = load_configuration(args)
    
    # Save configuration for reproducibility
    os.makedirs("configs", exist_ok=True)
    config_save_path = "configs/current_pace_config.yaml"
    import yaml
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    logging.info(f"Saved current configuration to {config_save_path}")
    
    try:
        # Prepare data
        train_loader, val_loader = prepare_data(config, args)
        
        # Create model
        model = create_model(config)
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Create trainer
        training_config = config["training"]
        trainer = SAETrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            device=args.device,
            log_dir=training_config["log_dir"],
            checkpoint_dir=training_config["checkpoint_dir"],
            use_wandb=training_config["use_wandb"],
            wandb_project=training_config["wandb_project"],
            l1_lambda=training_config["l1_lambda"],
            grad_clip_norm=training_config["grad_clip_norm"],
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
                num_epochs=training_config["num_epochs"],
                eval_every=training_config["eval_every"],
                save_every=training_config["save_every"],
                log_every=training_config["log_every"],
                max_steps=training_config.get("max_steps"),
            )
            
            logging.info("Training completed successfully!")
            
            # Final evaluation
            if val_loader is not None:
                logging.info("Running final evaluation...")
                final_metrics = trainer.evaluate(val_loader)
                
                logging.info("Final evaluation results:")
                for key, value in final_metrics.items():
                    logging.info(f"  {key}: {value:.4f}")
                    
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
        
    logging.info("Script completed successfully!")


if __name__ == "__main__":
    main()
 