"""Configuration classes for TopK SAE training.

This module defines configuration classes for all aspects of training,
including model architecture, training parameters, and data settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for TopK SAE model architecture.
    
    Args:
        input_dim: Dimension of input activations
        hidden_dim: Dimension of hidden layer (should be > input_dim for overcomplete)
        k: Number of top-k activations to keep
        tied_weights: Whether to tie encoder/decoder weights
        normalize_decoder: Whether to normalize decoder weights
        bias: Whether to include bias terms
        dtype: Data type for model parameters
        
    Example:
        >>> config = ModelConfig(
        ...     input_dim=2048,
        ...     hidden_dim=8192,
        ...     k=128
        ... )
    """
    input_dim: int
    hidden_dim: int
    k: int
    tied_weights: bool = True
    normalize_decoder: bool = True
    bias: bool = True
    dtype: str = "float32"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.k <= 0 or self.k > self.hidden_dim:
            raise ValueError(f"k must be in range (0, {self.hidden_dim}], got {self.k}")
        if self.hidden_dim <= self.input_dim:
            print(f"Warning: hidden_dim ({self.hidden_dim}) <= input_dim ({self.input_dim}). "
                  f"Consider using overcomplete representation (hidden_dim > input_dim)")


@dataclass
class DataConfig:
    """Configuration for data loading and processing.
    
    Args:
        model_name: Name of the language model
        layer_name: Name of layer to extract activations from
        dataset_name: Name of text dataset
        dataset_split: Dataset split to use
        num_samples: Number of samples to use (None for all)
        max_length: Maximum sequence length
        batch_size: Batch size for activation collection
        streaming: Whether to use streaming dataset
        cache_dir: Directory to cache model/data
        save_activations_path: Path to save collected activations
        
    Example:
        >>> config = DataConfig(
        ...     model_name="google/gemma-2-2b",
        ...     layer_name="model.layers.15",
        ...     dataset_name="openwebtext"
        ... )
    """
    model_name: str
    layer_name: str
    dataset_name: str = "openwebtext"
    dataset_split: str = "train"
    num_samples: Optional[int] = None
    max_length: int = 512
    batch_size: int = 8
    streaming: bool = False
    cache_dir: Optional[str] = None
    save_activations_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.layer_name:
            raise ValueError("layer_name cannot be empty")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ("adam", "adamw", "sgd")
        weight_decay: Weight decay for regularization
        l1_lambda: L1 regularization coefficient
        grad_clip_norm: Gradient clipping norm
        batch_size: Training batch size
        val_batch_size: Validation batch size
        train_val_split: Fraction of data for training
        scheduler: Learning rate scheduler type
        scheduler_params: Parameters for scheduler
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
        log_every: Log metrics every N steps
        max_steps: Maximum training steps
        early_stopping_patience: Early stopping patience
        use_wandb: Whether to use Weights & Biases
        wandb_project: WandB project name
        log_dir: Directory for logging
        checkpoint_dir: Directory for checkpoints
        
    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=10,
        ...     learning_rate=1e-3,
        ...     batch_size=128,
        ...     use_wandb=True
        ... )
    """
    num_epochs: int = 10
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    l1_lambda: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    batch_size: int = 128
    val_batch_size: int = 256
    train_val_split: float = 0.9
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    eval_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    max_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    use_wandb: bool = False
    wandb_project: str = "topk-sae"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(f"optimizer must be one of ['adam', 'adamw', 'sgd'], got {self.optimizer}")
        if not 0 < self.train_val_split < 1:
            raise ValueError(f"train_val_split must be in (0, 1), got {self.train_val_split}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.val_batch_size <= 0:
            raise ValueError(f"val_batch_size must be positive, got {self.val_batch_size}")


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs.
    
    Args:
        model: Model configuration
        data: Data configuration  
        training: Training configuration
        
    Example:
        >>> full_config = FullConfig(
        ...     model=ModelConfig(input_dim=2048, hidden_dim=8192, k=128),
        ...     data=DataConfig(model_name="google/gemma-2-2b", layer_name="model.layers.15"),
        ...     training=TrainingConfig(num_epochs=10, use_wandb=True)
        ... )
    """
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FullConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with 'model', 'data', 'training' keys
            
        Returns:
            FullConfig instance
        """
        return cls(
            model=ModelConfig(**config_dict["model"]),
            data=DataConfig(**config_dict["data"]),
            training=TrainingConfig(**config_dict["training"])
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__
        }
        
    def save_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file
        """
        import yaml
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        print(f"Configuration saved to {filepath}")
        
    @classmethod
    def load_yaml(cls, filepath: str) -> "FullConfig":
        """Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            FullConfig instance
        """
        import yaml
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)


# Default configurations for common use cases
def get_gemma2_2b_config() -> FullConfig:
    """Get default configuration for Gemma2-2b model.
    
    Returns:
        Default configuration for Gemma2-2b
    """
    return FullConfig(
        model=ModelConfig(
            input_dim=2304,  # Gemma2-2b hidden size
            hidden_dim=9216,  # 4x overcomplete
            k=128,  # ~5.5% sparsity
            tied_weights=True,
            normalize_decoder=True,
        ),
        data=DataConfig(
            model_name="google/gemma-2-2b",
            layer_name="model.layers.15",  # Middle layer
            dataset_name="openwebtext",
            max_length=512,
            batch_size=8,
            num_samples=100000,  # 100k samples
        ),
        training=TrainingConfig(
            num_epochs=5,
            learning_rate=1e-3,
            batch_size=256,
            l1_lambda=1e-4,
            eval_every=1000,
            save_every=5000,
            use_wandb=True,
            wandb_project="topk-sae-gemma2-2b",
        )
    )


def get_small_test_config() -> FullConfig:
    """Get configuration for small-scale testing.
    
    Returns:
        Configuration for quick testing
    """
    return FullConfig(
        model=ModelConfig(
            input_dim=512,
            hidden_dim=2048,
            k=32,
        ),
        data=DataConfig(
            model_name="google/gemma-2-2b",
            layer_name="model.layers.5",
            dataset_name="pyvene/axbench-concept10",
            max_length=256,
            batch_size=4,
            num_samples=1000,  # Small for testing
        ),
        training=TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            batch_size=32,
            eval_every=50,
            save_every=100,
            use_wandb=False,
        )
    ) 