"""Absolute TopK Sparse Autoencoder implementation for neural network interpretation.

This module implements an Absolute TopK Sparse Autoencoder that constrains the number of
active neurons during reconstruction based on absolute values, promoting sparsity and interpretability.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AbsoluteTopKSAE(nn.Module):
    """Absolute TopK Sparse Autoencoder for learning sparse representations.
    
    The Absolute TopK SAE constrains the number of active neurons to exactly k during
    the forward pass by selecting the top-k absolute values, promoting sparse and interpretable representations.
    
    Args:
        input_dim: Dimension of input activations
        hidden_dim: Dimension of the hidden layer (typically much larger than input_dim)
        k: Number of top absolute value activations to keep (sparsity parameter)
        tied_weights: Whether to tie encoder and decoder weights
        normalize_decoder: Whether to normalize decoder weights to unit norm
        bias: Whether to include bias terms
        device: Device to place the model on
        dtype: Data type for model parameters
        
    Example:
        >>> sae = AbsoluteTopKSAE(input_dim=512, hidden_dim=2048, k=64)
        >>> x = torch.randn(32, 512)  # batch_size=32, input_dim=512
        >>> reconstruction, hidden_acts, loss_dict = sae(x)
        >>> print(f"Reconstruction shape: {reconstruction.shape}")
        >>> print(f"Sparsity: {(hidden_acts != 0).float().mean():.3f}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        tied_weights: bool = True,
        normalize_decoder: bool = True,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        
        # Validate arguments
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if k <= 0 or k > hidden_dim:
            raise ValueError(f"k must be in range (0, {hidden_dim}], got {k}")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        
        # Set device and dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        
        # Initialize encoder
        self.encoder = nn.Linear(
            input_dim, hidden_dim, bias=bias, device=device, dtype=dtype
        )
        
        # Initialize decoder
        if tied_weights:
            # Use transpose of encoder weights
            self.decoder_weight = None  # Will use encoder.weight.T
            if bias:
                self.decoder_bias = nn.Parameter(
                    torch.zeros(input_dim, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("decoder_bias", None)
        else:
            self.decoder = nn.Linear(
                hidden_dim, input_dim, bias=bias, device=device, dtype=dtype
            )
            
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        
        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            
        # Initialize biases to zero
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if hasattr(self, "decoder") and self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        if hasattr(self, "decoder_bias") and self.decoder_bias is not None:
            nn.init.zeros_(self.decoder_bias)
            
    def _normalize_decoder_weights(self) -> None:
        """Normalize decoder weights to unit norm along input dimension."""
        if self.normalize_decoder:
            if self.tied_weights:
                # Normalize encoder weights (which are used as decoder weights)
                with torch.no_grad():
                    self.encoder.weight.data = F.normalize(
                        self.encoder.weight.data, dim=0
                    )
            else:
                with torch.no_grad():
                    self.decoder.weight.data = F.normalize(
                        self.decoder.weight.data, dim=1
                    )
                    
    def _get_decoder_weight(self) -> Tensor:
        """Get decoder weight matrix."""
        if self.tied_weights:
            return self.encoder.weight.T
        else:
            return self.decoder.weight
            
    def _get_decoder_bias(self) -> Optional[Tensor]:
        """Get decoder bias vector."""
        if self.tied_weights:
            return self.decoder_bias
        else:
            return self.decoder.bias
            
    def encode(self, x: Tensor) -> Tensor:
        """Encode input to hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Hidden activations of shape (batch_size, hidden_dim)
        """
        return self.encoder(x)
        
    def apply_topk(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply Absolute TopK sparsity to hidden activations.
        
        Args:
            hidden: Hidden activations of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of:
                - sparse_hidden: TopK sparse activations
                - topk_indices: Indices of top-k activations
        """
        # Get top-k values and indices based on absolute values
        topk_values, topk_indices = torch.topk(
            hidden.abs(), k=self.k, dim=-1, largest=True, sorted=False
        )
        
        # Create a sparse tensor with original values at top-k indices
        sparse_hidden = torch.zeros_like(hidden)
        # Use gather to select the original values at the topk_indices
        original_values = hidden.gather(-1, topk_indices)
        sparse_hidden.scatter_(-1, topk_indices, original_values)
        
        return sparse_hidden, topk_indices
        
    def decode(self, sparse_hidden: Tensor) -> Tensor:
        """Decode sparse hidden representation to reconstruction.
        
        Args:
            sparse_hidden: Sparse hidden activations
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        decoder_weight = self._get_decoder_weight()
        decoder_bias = self._get_decoder_bias()
        
        reconstruction = F.linear(sparse_hidden, decoder_weight, decoder_bias)
        return reconstruction
        
    def forward(
        self, x: Tensor, return_aux_losses: bool = True
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Forward pass through the Absolute TopK SAE.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_aux_losses: Whether to compute auxiliary losses
            
        Returns:
            Tuple of:
                - reconstruction: Reconstructed input
                - sparse_hidden: Sparse hidden activations
                - loss_dict: Dictionary containing various loss components
        """
        # Normalize decoder weights if required
        if self.training:
            self._normalize_decoder_weights()
            
        # Encode
        hidden = self.encode(x)
        
        # Apply TopK sparsity
        sparse_hidden, topk_indices = self.apply_topk(hidden)
        
        # Decode
        reconstruction = self.decode(sparse_hidden)
        
        # Compute losses
        loss_dict = {}
        if return_aux_losses:
            loss_dict = self._compute_losses(
                x, reconstruction, hidden, sparse_hidden, topk_indices
            )
            
        return reconstruction, sparse_hidden, loss_dict
        
    def _compute_losses(
        self,
        x: Tensor,
        reconstruction: Tensor,
        hidden: Tensor,
        sparse_hidden: Tensor,
        topk_indices: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute various loss components.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            hidden: Dense hidden activations
            sparse_hidden: Sparse hidden activations
            topk_indices: Indices of top-k activations
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
        
        # Sparsity statistics
        num_active = (sparse_hidden != 0).float().sum(dim=-1).mean()
        sparsity_ratio = num_active / self.hidden_dim
        
        # L1 penalty on sparse activations
        l1_loss = sparse_hidden.abs().mean()
        
        # Auxiliary losses for analysis
        loss_dict = {
            "reconstruction_loss": recon_loss,
            "l1_loss": l1_loss,
            "num_active": num_active,
            "sparsity_ratio": sparsity_ratio,
            "mean_activation": sparse_hidden.mean(),
            "max_activation": sparse_hidden.max(),
        }
        
        return loss_dict
        
    def get_feature_density(self, dataloader: torch.utils.data.DataLoader) -> Tensor:
        """Compute feature density (activation frequency) across a dataset.
        
        Args:
            dataloader: DataLoader yielding input tensors
            
        Returns:
            Feature density tensor of shape (hidden_dim,)
        """
        self.eval()
        feature_counts = torch.zeros(self.hidden_dim, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                x = x.to(self.device)
                _, sparse_hidden, _ = self.forward(x, return_aux_losses=False)
                
                # Count active features
                active_features = (sparse_hidden != 0).float().sum(dim=0)
                feature_counts += active_features
                total_samples += x.shape[0]
                
        feature_density = feature_counts / total_samples
        return feature_density
        
    def get_reconstruction_metrics(
        self, x: Tensor
    ) -> Dict[str, Union[float, Tensor]]:
        """Compute detailed reconstruction metrics.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing reconstruction metrics
        """
        self.eval()
        with torch.no_grad():
            reconstruction, sparse_hidden, loss_dict = self.forward(x)
            
            # Additional metrics
            cosine_sim = F.cosine_similarity(x, reconstruction, dim=-1).mean()
            explained_variance = 1 - torch.var(x - reconstruction) / torch.var(x)
            
            metrics = {
                "mse": loss_dict["reconstruction_loss"].item(),
                "cosine_similarity": cosine_sim.item(),
                "explained_variance": explained_variance.item(),
                "sparsity": loss_dict["sparsity_ratio"].item(),
                "l1_norm": loss_dict["l1_loss"].item(),
            }
            
        return metrics
        
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "tied_weights": self.tied_weights,
            "normalize_decoder": self.normalize_decoder,
        }
        torch.save(checkpoint, filepath)
        
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[str] = None) -> "AbsoluteTopKSAE":
        """Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded AbsoluteTopKSAE model
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            k=checkpoint["k"],
            tied_weights=checkpoint["tied_weights"],
            normalize_decoder=checkpoint["normalize_decoder"],
            device=device,
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
