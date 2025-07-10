"""Data loading utilities for PaCE-1M concept dataset.

This module provides utilities for loading and processing the PaCE-1M concept
dataset for TopK SAE training.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm


class PaCEConceptDataset(Dataset):
    """Dataset for PaCE-1M concept representations.
    
    This dataset loads concept representations from the PaCE-1M dataset,
    which contains concept-specific activation patterns for training
    interpretable sparse autoencoders.
    
    Args:
        concept_dir: Directory containing concept files
        concept_index_path: Path to concept index file
        max_concepts: Maximum number of concepts to load (None for all)
        normalize: Whether to normalize activations
        
    Example:
        >>> dataset = PaCEConceptDataset("./concept", "./concept_index.txt")
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        concept_dir: str,
        concept_index_path: str,
        max_concepts: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.concept_dir = Path(concept_dir)
        self.concept_index_path = Path(concept_index_path)
        self.normalize = normalize
        
        # Load concept index
        self.concept_list = self._load_concept_index()
        
        # Limit concepts if specified
        if max_concepts is not None:
            self.concept_list = self.concept_list[:max_concepts]
            
        # Load concept representations
        self.concept_representations = self._load_concept_representations()
        
        print(f"Loaded {len(self.concept_representations)} concept representations")
        print(f"Representation dimension: {self.concept_representations.shape[1]}")
        
    def _load_concept_index(self) -> List[str]:
        """Load concept index file.
        
        Returns:
            List of concept names in ranked order
        """
        if not self.concept_index_path.exists():
            raise FileNotFoundError(f"Concept index file not found: {self.concept_index_path}")
            
        with open(self.concept_index_path, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(concepts)} concepts from index file")
        return concepts
        
    def _load_concept_representations(self) -> torch.Tensor:
        """Load concept representations from individual files.
        
        Returns:
            Tensor of concept representations
        """
        if not self.concept_dir.exists():
            raise FileNotFoundError(f"Concept directory not found: {self.concept_dir}")
            
        representations = []
        valid_concepts = []
        
        print("Loading concept representations...")
        for concept in tqdm(self.concept_list, desc="Loading concepts"):
            concept_file = self.concept_dir / f"{concept}.npy"
            
            if concept_file.exists():
                try:
                    # Load numpy array
                    concept_repr = np.load(concept_file)
                    
                    # Convert to tensor
                    if isinstance(concept_repr, np.ndarray):
                        concept_tensor = torch.from_numpy(concept_repr).float()
                    else:
                        concept_tensor = torch.tensor(concept_repr, dtype=torch.float32)
                        
                    # Ensure 1D representation
                    if concept_tensor.dim() > 1:
                        concept_tensor = concept_tensor.flatten()
                        
                    representations.append(concept_tensor)
                    valid_concepts.append(concept)
                    
                except Exception as e:
                    print(f"Warning: Failed to load concept {concept}: {e}")
                    continue
            else:
                print(f"Warning: Concept file not found: {concept_file}")
                
        if not representations:
            raise ValueError("No valid concept representations found")
            
        # Stack representations
        concept_tensor = torch.stack(representations)
        
        # Normalize if requested
        if self.normalize:
            concept_tensor = torch.nn.functional.normalize(concept_tensor, dim=1)
            
        # Update concept list to only include valid concepts
        self.concept_list = valid_concepts
        
        return concept_tensor
        
    def __len__(self) -> int:
        return len(self.concept_representations)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get concept representation by index.
        
        Args:
            idx: Index of concept
            
        Returns:
            Concept representation tensor
        """
        return self.concept_representations[idx]
        
    def get_concept_name(self, idx: int) -> str:
        """Get concept name by index.
        
        Args:
            idx: Index of concept
            
        Returns:
            Concept name
        """
        return self.concept_list[idx]
        
    def get_concept_names(self) -> List[str]:
        """Get all concept names.
        
        Returns:
            List of concept names
        """
        return self.concept_list.copy()
        
    def get_representation_dim(self) -> int:
        """Get dimension of concept representations.
        
        Returns:
            Dimension of representations
        """
        return self.concept_representations.shape[1]


class PaCEConceptDataLoader:
    """High-level interface for loading PaCE-1M concept data.
    
    This class provides utilities for loading and processing the PaCE-1M
    concept dataset for SAE training.
    
    Args:
        concept_dir: Directory containing concept files
        concept_index_path: Path to concept index file
        cache_dir: Directory to cache processed data
        
    Example:
        >>> loader = PaCEConceptDataLoader("./concept", "./concept_index.txt")
        >>> train_dataset = loader.create_dataset(max_concepts=1000)
    """
    
    def __init__(
        self,
        concept_dir: str,
        concept_index_path: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.concept_dir = concept_dir
        self.concept_index_path = concept_index_path
        self.cache_dir = cache_dir
        
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
    def download_and_extract_dataset(self, output_dir: str = "./") -> None:
        """Download and extract the PaCE-1M dataset.
        
        Args:
            output_dir: Directory to extract dataset to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download concept.zip and concept_index.txt from GitHub
        import urllib.request
        
        base_url = "https://raw.githubusercontent.com/peterljq/Parsimonious-Concept-Engineering/main"
        
        print("Downloading PaCE-1M dataset...")
        
        # Download concept_index.txt
        index_url = f"{base_url}/concept_index.txt"
        index_path = output_path / "concept_index.txt"
        
        print(f"Downloading concept index to {index_path}")
        urllib.request.urlretrieve(index_url, index_path)
        
        # Download concept.zip
        zip_url = f"{base_url}/concept.zip"
        zip_path = output_path / "concept.zip"
        
        print(f"Downloading concept data to {zip_path}")
        urllib.request.urlretrieve(zip_url, zip_path)
        
        # Extract concept.zip
        print("Extracting concept data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
            
        print("Dataset download and extraction completed!")
        print(f"Concept index: {index_path}")
        print(f"Concept directory: {output_path / 'concept'}")
        
    def create_dataset(
        self,
        max_concepts: Optional[int] = None,
        normalize: bool = True,
        train_val_split: float = 0.8,
        random_seed: int = 42,
    ) -> Tuple[PaCEConceptDataset, PaCEConceptDataset]:
        """Create train/validation datasets from PaCE-1M.
        
        Args:
            max_concepts: Maximum number of concepts to use
            normalize: Whether to normalize representations
            train_val_split: Fraction of data for training
            random_seed: Random seed for splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create full dataset
        full_dataset = PaCEConceptDataset(
            concept_dir=self.concept_dir,
            concept_index_path=self.concept_index_path,
            max_concepts=max_concepts,
            normalize=normalize,
        )
        
        # Split into train/validation
        dataset_size = len(full_dataset)
        train_size = int(train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        # Set random seed for reproducible splits
        torch.manual_seed(random_seed)
        
        train_indices, val_indices = torch.utils.data.random_split(
            range(dataset_size), [train_size, val_size]
        )
        
        # Create train dataset
        train_concepts = [full_dataset.get_concept_name(i) for i in train_indices.indices]
        train_dataset = PaCEConceptDataset(
            concept_dir=self.concept_dir,
            concept_index_path=self.concept_index_path,
            max_concepts=max_concepts,
            normalize=normalize,
        )
        # Filter to only train concepts
        train_dataset.concept_list = train_concepts
        train_dataset.concept_representations = full_dataset.concept_representations[train_indices.indices]
        
        # Create validation dataset
        val_concepts = [full_dataset.get_concept_name(i) for i in val_indices.indices]
        val_dataset = PaCEConceptDataset(
            concept_dir=self.concept_dir,
            concept_index_path=self.concept_index_path,
            max_concepts=max_concepts,
            normalize=normalize,
        )
        # Filter to only validation concepts
        val_dataset.concept_list = val_concepts
        val_dataset.concept_representations = full_dataset.concept_representations[val_indices.indices]
        
        print(f"Train dataset: {len(train_dataset)} concepts")
        print(f"Validation dataset: {len(val_dataset)} concepts")
        
        return train_dataset, val_dataset
        
    def create_dataloaders(
        self,
        max_concepts: Optional[int] = None,
        batch_size: int = 32,
        train_val_split: float = 0.8,
        num_workers: int = 2,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation dataloaders.
        
        Args:
            max_concepts: Maximum number of concepts to use
            batch_size: Batch size for training
            train_val_split: Fraction of data for training
            num_workers: Number of workers for data loading
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset, val_dataset = self.create_dataset(
            max_concepts=max_concepts,
            train_val_split=train_val_split,
            **kwargs
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batches for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        return train_loader, val_loader
        
    def analyze_concepts(self, dataset: PaCEConceptDataset) -> Dict[str, Union[float, int]]:
        """Analyze concept dataset statistics.
        
        Args:
            dataset: PaCE concept dataset
            
        Returns:
            Dictionary of analysis results
        """
        representations = dataset.concept_representations
        
        analysis = {
            "num_concepts": len(dataset),
            "representation_dim": dataset.get_representation_dim(),
            "mean_norm": representations.norm(dim=1).mean().item(),
            "std_norm": representations.norm(dim=1).std().item(),
            "mean_activation": representations.mean().item(),
            "std_activation": representations.std().item(),
            "max_activation": representations.max().item(),
            "min_activation": representations.min().item(),
        }
        
        return analysis


def create_pace_config() -> Dict[str, any]:
    """Create configuration for PaCE-1M training.
    
    Returns:
        Configuration dictionary for PaCE-1M training
    """
    return {
        "model": {
            "input_dim": 4096,  # Typical LLaMA-2 hidden size
            "hidden_dim": 16384,  # 4x overcomplete
            "k": 512,  # ~3% sparsity
            "tied_weights": True,
            "normalize_decoder": True,
            "bias": True,
            "dtype": "float32",
        },
        "data": {
            "concept_dir": "./concept",
            "concept_index_path": "./concept_index.txt",
            "max_concepts": 10000,  # Use first 10k concepts
            "normalize": True,
            "train_val_split": 0.8,
        },
        "training": {
            "num_epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "l1_lambda": 0.0001,
            "grad_clip_norm": 1.0,
            "batch_size": 128,
            "val_batch_size": 256,
            "eval_every": 500,
            "save_every": 2000,
            "log_every": 100,
            "use_wandb": True,
            "wandb_project": "topk-sae-pace",
            "log_dir": "logs/pace",
            "checkpoint_dir": "checkpoints/pace",
        }
    } 