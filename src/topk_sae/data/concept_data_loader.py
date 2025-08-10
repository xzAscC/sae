"""Data loading utilities for concept-based datasets.

This module provides utilities for loading concept-based datasets for SAE training.
"""

import os
import torch
from torch.utils.data import Dataset
import requests
import tarfile
from tqdm import tqdm
from typing import Optional, Union, Tuple

class ConceptDataset(Dataset):
    """Dataset for loading concept-based activations.
    
    Args:
        activations: Tensor of shape (num_samples, activation_dim)
        labels: Optional labels for the activations
    """
    
    def __init__(self, activations: torch.Tensor, labels: Optional[torch.Tensor] = None):
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

class ConceptDatasetLoader:
    """Loads concept-based datasets from a URL.
    
    Args:
        url: URL to the dataset
        cache_dir: Directory to cache the dataset
    """
    
    def __init__(self, url: str, cache_dir: str = "./cache"):
        self.url = url
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _download_and_extract(self) -> str:
        """Download and extract the dataset.
        
        Returns:
            Path to the extracted dataset
        """
        filename = self.url.split("/")[-1]
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading dataset from {self.url}...")
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            
            with open(filepath, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
        
        extract_path = os.path.join(self.cache_dir, "concept_dataset")
        if not os.path.exists(extract_path):
            print(f"Extracting {filepath}...")
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=self.cache_dir)
            # Rename the extracted folder to a consistent name
            extracted_folder = tar.getnames()[0].split('/')[0]
            os.rename(os.path.join(self.cache_dir, extracted_folder), extract_path)
            
        return extract_path
        
    def load_dataset(self) -> ConceptDataset:
        """Load the dataset from the extracted files.
        
        Returns:
            ConceptDataset instance
        """
        extract_path = self._download_and_extract()
        
        # Assuming the dataset contains .pt files for activations
        activation_files = [f for f in os.listdir(extract_path) if f.endswith(".pt")]
        
        if not activation_files:
            raise FileNotFoundError(f"No .pt files found in {extract_path}")
            
        # Load and concatenate all activations
        all_activations = []
        for filename in activation_files:
            filepath = os.path.join(extract_path, filename)
            activations = torch.load(filepath, map_location="cpu")
            all_activations.append(activations)
            
        concatenated_activations = torch.cat(all_activations, dim=0)
        
        return ConceptDataset(concatenated_activations)