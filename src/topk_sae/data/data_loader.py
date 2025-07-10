"""Data loading utilities for collecting neural network activations.

This module provides utilities for collecting activations from language models
using nnsight and preparing them for SAE training.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm


class ActivationDataset(Dataset):
    """Dataset for storing and accessing neural network activations.
    
    Args:
        activations: Tensor of shape (num_samples, activation_dim)
        labels: Optional labels for the activations
        
    Example:
        >>> activations = torch.randn(1000, 512)
        >>> dataset = ActivationDataset(activations)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        activations: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self.activations = activations
        self.labels = labels
        
        if labels is not None and len(activations) != len(labels):
            raise ValueError(
                f"Activations and labels must have same length, "
                f"got {len(activations)} and {len(labels)}"
            )
            
    def __len__(self) -> int:
        return len(self.activations)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.labels is not None:
            return self.activations[idx], self.labels[idx]
        return self.activations[idx]


class StreamingActivationDataset(IterableDataset):
    """Streaming dataset for generating activations on-the-fly.
    
    This dataset generates activations from a language model as needed,
    without storing all activations in memory.
    
    Args:
        model: nnsight LanguageModel instance
        tokenizer: Tokenizer for the model
        text_dataset: Dataset containing text data
        layer_name: Name of the layer to extract activations from
        max_length: Maximum sequence length
        batch_size: Batch size for text processing
        
    Example:
        >>> model = LanguageModel("google/gemma-2-2b", device_map="auto")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        >>> dataset = load_dataset("openwebtext", split="train", streaming=True)
        >>> activation_dataset = StreamingActivationDataset(
        ...     model, tokenizer, dataset, "model.layers.15", max_length=512
        ... )
    """
    
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: PreTrainedTokenizer,
        text_dataset: Any,
        layer_name: str,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.text_dataset = text_dataset
        self.layer_name = layer_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _process_text_batch(self, texts: List[str]) -> torch.Tensor:
        """Process a batch of texts and extract activations.
        
        Args:
            texts: List of text strings
            
        Returns:
            Activations tensor of shape (batch_size * seq_len, hidden_dim)
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Extract activations using nnsight
        with self.model.trace(inputs["input_ids"]) as tracer:
            # Get the specified layer
            layer = self.model
            for part in self.layer_name.split("."):
                layer = getattr(layer, part)
            
            # Save the layer output
            activations = layer.output.save()
            
        # Get activations and reshape
        activations_tensor = activations.value
        if len(activations_tensor.shape) == 3:  # (batch, seq, hidden)
            # Flatten sequence dimension
            batch_size, seq_len, hidden_dim = activations_tensor.shape
            activations_tensor = activations_tensor.reshape(-1, hidden_dim)
            
        return activations_tensor
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over the dataset, yielding activation tensors."""
        text_batch = []
        
        for item in self.text_dataset:
            # Extract text from dataset item
            if isinstance(item, dict):
                text = item.get("text", str(item))
            else:
                text = str(item)
                
            text_batch.append(text)
            
            # Process batch when it's full
            if len(text_batch) >= self.batch_size:
                try:
                    activations = self._process_text_batch(text_batch)
                    # Yield individual activation vectors
                    for i in range(activations.shape[0]):
                        yield activations[i]
                except Exception as e:
                    print(f"Warning: Failed to process batch: {e}")
                    
                text_batch = []
                
        # Process remaining texts
        if text_batch:
            try:
                activations = self._process_text_batch(text_batch)
                for i in range(activations.shape[0]):
                    yield activations[i]
            except Exception as e:
                print(f"Warning: Failed to process final batch: {e}")


class ActivationDataLoader:
    """High-level interface for loading activation data for SAE training.
    
    This class provides utilities for collecting activations from language models
    and preparing them for SAE training.
    
    Args:
        model_name: Name of the model to load
        layer_name: Name of the layer to extract activations from
        device: Device to run the model on
        cache_dir: Directory to cache model weights
        
    Example:
        >>> loader = ActivationDataLoader(
        ...     model_name="google/gemma-2-2b",
        ...     layer_name="model.layers.15"
        ... )
        >>> train_dataset = loader.create_dataset_from_text(
        ...     dataset_name="openwebtext",
        ...     split="train",
        ...     num_samples=10000
        ... )
    """
    
    def __init__(
        self,
        model_name: str,
        layer_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.layer_name = layer_name
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model {self.model_name}...")
        
        # Load model with nnsight
        self.model = LanguageModel(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            cache_dir=self.cache_dir,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded on device: {self.model.device}")
        
    def get_activation_dim(self) -> int:
        """Get the dimension of activations from the specified layer.
        
        Returns:
            Dimension of the activation vectors
        """
        # Create a dummy input to get activation shape
        dummy_input = self.tokenizer(
            "Hello world",
            return_tensors="pt",
            padding=True,
            max_length=10,
        )
        
        dummy_input = {k: v.to(self.model.device) for k, v in dummy_input.items()}
        
        with self.model.trace(dummy_input["input_ids"]) as tracer:
            # Get the specified layer
            layer = self.model
            for part in self.layer_name.split("."):
                layer = getattr(layer, part)
            
            # Save the layer output
            activations = layer.output.save()
            
        activation_shape = activations.value.shape
        return activation_shape[-1]  # Hidden dimension
        
    def collect_activations(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Collect activations from a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
            
        Returns:
            Tensor of activations of shape (total_tokens, hidden_dim)
        """
        all_activations = []
        
        # Process texts in batches
        batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]
        
        if show_progress:
            batches = tqdm(batches, desc="Collecting activations")
            
        for batch_texts in batches:
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Extract activations
            with self.model.trace(inputs["input_ids"]) as tracer:
                # Get the specified layer
                layer = self.model
                for part in self.layer_name.split("."):
                    layer = getattr(layer, part)
                
                # Save the layer output
                activations = layer.output.save()
                
            # Process activations
            activations_tensor = activations.value
            if len(activations_tensor.shape) == 3:  # (batch, seq, hidden)
                # Flatten sequence dimension
                batch_size, seq_len, hidden_dim = activations_tensor.shape
                activations_tensor = activations_tensor.reshape(-1, hidden_dim)
                
            all_activations.append(activations_tensor.cpu())
            
        # Concatenate all activations
        return torch.cat(all_activations, dim=0)
        
    def create_dataset_from_text(
        self,
        dataset_name: str,
        split: str = "train",
        num_samples: Optional[int] = None,
        streaming: bool = False,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> Union[ActivationDataset, StreamingActivationDataset]:
        """Create an activation dataset from a text dataset.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to use
            num_samples: Number of samples to use (None for all)
            streaming: Whether to use streaming dataset
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            ActivationDataset or StreamingActivationDataset
        """
        # Load text dataset
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        if streaming:
            return StreamingActivationDataset(
                model=self.model,
                tokenizer=self.tokenizer,
                text_dataset=dataset,
                layer_name=self.layer_name,
                max_length=max_length,
                batch_size=batch_size,
            )
        else:
            # Collect texts
            texts = []
            for i, item in enumerate(dataset):
                if num_samples is not None and i >= num_samples:
                    break
                    
                if isinstance(item, dict):
                    text = item.get("text", str(item))
                else:
                    text = str(item)
                texts.append(text)
                
            # Collect activations
            activations = self.collect_activations(
                texts, batch_size=batch_size, max_length=max_length
            )
            
            return ActivationDataset(activations)
            
    def save_activations(
        self, activations: torch.Tensor, filepath: str
    ) -> None:
        """Save activations to disk.
        
        Args:
            activations: Activation tensor to save
            filepath: Path to save the activations
        """
        torch.save(activations, filepath)
        print(f"Saved {activations.shape[0]} activations to {filepath}")
        
    def load_activations(self, filepath: str) -> torch.Tensor:
        """Load activations from disk.
        
        Args:
            filepath: Path to the saved activations
            
        Returns:
            Loaded activation tensor
        """
        activations = torch.load(filepath, map_location="cpu")
        print(f"Loaded {activations.shape[0]} activations from {filepath}")
        return activations 