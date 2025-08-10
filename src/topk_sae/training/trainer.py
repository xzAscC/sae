"""Training utilities for TopK Sparse Autoencoders.

This module provides a comprehensive trainer for TopK SAE models with
logging, checkpointing, and evaluation capabilities.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb

from ..models.topk_sae import TopKSAE
from ..data.data_loader import ActivationDataset


class SAETrainer:
    """Trainer for TopK Sparse Autoencoders.
    
    This trainer provides comprehensive training functionality including:
    - Loss computation and optimization
    - Learning rate scheduling
    - Checkpointing and model saving
    - Logging with TensorBoard and WandB
    - Evaluation and metrics computation
    
    Args:
        model: TopK SAE model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Optional learning rate scheduler
        device: Device to train on
        log_dir: Directory for logging
        checkpoint_dir: Directory for saving checkpoints
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        
    Example:
        >>> model = TopKSAE(input_dim=2048, hidden_dim=8192, k=128)
        >>> trainer = SAETrainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        ...     use_wandb=True,
        ...     wandb_project="topk-sae-gemma2"
        ... )
        >>> trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: TopKSAE,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "topk-sae",
        l1_lambda: float = 1e-4,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.l1_lambda = l1_lambda
        self.grad_clip_norm = grad_clip_norm
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        
        # Set up directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, config=self._get_config())
            
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": [],
            "sparsity": [],
            "learning_rate": [],
        }
        
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for logging."""
        return {
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "k": self.model.k,
                "tied_weights": self.model.tied_weights,
                "normalize_decoder": self.model.normalize_decoder,
            },
            "training_config": {
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "l1_lambda": self.l1_lambda,
                "grad_clip_norm": self.grad_clip_norm,
                "device": str(self.device),
            },
        }
        
    def compute_loss(
        self, reconstruction: torch.Tensor, target: torch.Tensor, loss_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total training loss.
        
        Args:
            reconstruction: Reconstructed activations
            target: Target activations
            loss_dict: Dictionary of auxiliary losses from model
            
        Returns:
            Tuple of total loss and loss components dictionary
        """
        # Reconstruction loss
        recon_loss = loss_dict["reconstruction_loss"]
        
        # L1 sparsity penalty
        l1_loss = loss_dict["l1_loss"]
        
        # Total loss
        total_loss = recon_loss + self.l1_lambda * l1_loss
        
        # Loss components for logging
        loss_components = {
            "total_loss": total_loss.item(),
            "reconstruction_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "sparsity_ratio": loss_dict["sparsity_ratio"].item(),
            "num_active": loss_dict["num_active"].item(),
            "mean_activation": loss_dict["mean_activation"].item(),
            "max_activation": loss_dict["max_activation"].item(),
        }
        
        return total_loss, loss_components
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Batch of activation data
            
        Returns:
            Dictionary of loss components
        """
        self.model.train()
        
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Get activations if (activations, labels) tuple
        batch = batch.to(self.device)
        
        # Forward pass
        reconstruction, sparse_hidden, loss_dict = self.model(batch)
        
        # Compute loss
        total_loss, loss_components = self.compute_loss(reconstruction, batch, loss_dict)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
            loss_components["grad_norm"] = grad_norm.item()
            
        # Optimizer step
        self.optimizer.step()
        
        return loss_components
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        total_sparsity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                
                # Forward pass
                reconstruction, sparse_hidden, loss_dict = self.model(batch)
                
                # Compute loss
                total_loss_batch, loss_components = self.compute_loss(
                    reconstruction, batch, loss_dict
                )
                
                total_loss += loss_components["total_loss"]
                total_recon_loss += loss_components["reconstruction_loss"]
                total_l1_loss += loss_components["l1_loss"]
                total_sparsity += loss_components["sparsity_ratio"]
                num_batches += 1
                
        # Average metrics
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_recon_loss": total_recon_loss / num_batches,
            "val_l1_loss": total_l1_loss / num_batches,
            "val_sparsity": total_sparsity / num_batches,
        }
        
        return metrics
        
    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """Log metrics to TensorBoard and WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step number
            prefix: Prefix for metric names
        """
        for key, value in metrics.items():
            # TensorBoard logging
            self.writer.add_scalar(f"{prefix}{key}", value, step)
            
            # WandB logging
            if self.use_wandb:
                wandb.log({f"{prefix}{key}": value}, step=step)
                
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        val_loss: float,
        is_best: bool = False,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            step: Current step
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            extra_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "k": self.model.k,
                "tied_weights": self.model.tied_weights,
                "normalize_decoder": self.model.normalize_decoder,
            },
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if extra_info is not None:
            checkpoint.update(extra_info)
            
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        return str(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["step"]
        self.best_val_loss = checkpoint["val_loss"]
        
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]
            
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
        return checkpoint
        
    def train(
        self,
        num_epochs: int,
        eval_every: int = 1000,
        save_every: int = 5000,
        log_every: int = 100,
        max_steps: Optional[int] = None,
    ) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of training epochs
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            log_every: Log metrics every N steps
            max_steps: Maximum number of training steps
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model: {self.model.input_dim} -> {self.model.hidden_dim} (k={self.model.k})")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            print(f"Val batches: {len(self.val_dataloader)}")
            
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_sparsity = 0.0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            )
            
            for batch in pbar:
                # Training step
                loss_components = self.train_step(batch)
                
                # Update epoch metrics
                epoch_loss += loss_components["total_loss"]
                epoch_recon_loss += loss_components["reconstruction_loss"]
                epoch_sparsity += loss_components["sparsity_ratio"]
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_components['total_loss']:.4f}",
                    "recon": f"{loss_components['reconstruction_loss']:.4f}",
                    "sparse": f"{loss_components['sparsity_ratio']:.3f}",
                })
                
                # Logging
                if self.global_step % log_every == 0:
                    # Add learning rate to metrics
                    loss_components["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                    self.log_metrics(loss_components, self.global_step, "train/")
                    
                # Evaluation
                if self.val_dataloader and self.global_step % eval_every == 0:
                    val_metrics = self.evaluate(self.val_dataloader)
                    self.log_metrics(val_metrics, self.global_step, "val/")
                    
                    # Check for best model
                    val_loss = val_metrics["val_loss"]
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        
                    print(f"\nValidation at step {self.global_step}:")
                    print(f"  Val Loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")
                    print(f"  Val Recon: {val_metrics['val_recon_loss']:.4f}")
                    print(f"  Val Sparsity: {val_metrics['val_sparsity']:.3f}")
                    
                # Checkpointing
                if self.global_step % save_every == 0:
                    val_loss = self.best_val_loss
                    if self.val_dataloader:
                        val_metrics = self.evaluate(self.val_dataloader)
                        val_loss = val_metrics["val_loss"]
                        
                    checkpoint_path = self.save_checkpoint(
                        epoch=epoch,
                        step=self.global_step,
                        val_loss=val_loss,
                        is_best=val_loss < self.best_val_loss,
                    )
                    print(f"\nSaved checkpoint: {checkpoint_path}")
                    
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                self.global_step += 1
                
                # Early stopping
                if max_steps is not None and self.global_step >= max_steps:
                    print(f"\nReached maximum steps ({max_steps}). Stopping training.")
                    return
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_recon = epoch_recon_loss / num_batches
            avg_epoch_sparsity = epoch_sparsity / num_batches
            
            # Update training history
            self.training_history["train_loss"].append(avg_epoch_loss)
            self.training_history["train_recon_loss"].append(avg_epoch_recon)
            self.training_history["sparsity"].append(avg_epoch_sparsity)
            self.training_history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            
            # Epoch-level validation
            if self.val_dataloader:
                val_metrics = self.evaluate(self.val_dataloader)
                self.training_history["val_loss"].append(val_metrics["val_loss"])
                self.training_history["val_recon_loss"].append(val_metrics["val_recon_loss"])
                
                # Log epoch-level metrics
                epoch_metrics = {
                    "epoch_train_loss": avg_epoch_loss,
                    "epoch_val_loss": val_metrics["val_loss"],
                    "epoch_train_recon": avg_epoch_recon,
                    "epoch_val_recon": val_metrics["val_recon_loss"],
                    "epoch_sparsity": avg_epoch_sparsity,
                }
                self.log_metrics(epoch_metrics, epoch, "epoch/")
                
            print(f"\nEpoch {epoch + 1} completed:")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Train Recon: {avg_epoch_recon:.4f}")
            print(f"  Sparsity: {avg_epoch_sparsity:.3f}")
            if self.val_dataloader:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                
        # Final checkpoint
        final_val_loss = self.best_val_loss
        if self.val_dataloader:
            val_metrics = self.evaluate(self.val_dataloader)
            final_val_loss = val_metrics["val_loss"]
            
        final_checkpoint = self.save_checkpoint(
            epoch=num_epochs,
            step=self.global_step,
            val_loss=final_val_loss,
            is_best=final_val_loss < self.best_val_loss,
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final checkpoint: {final_checkpoint}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
            
    def analyze_features(
        self, dataloader: DataLoader, num_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze learned features.
        
        Args:
            dataloader: DataLoader for analysis
            num_batches: Number of batches to analyze (None for all)
            
        Returns:
            Dictionary containing feature analysis results
        """
        self.model.eval()
        
        # Collect feature activations
        all_sparse_hidden = []
        all_reconstructions = []
        all_targets = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                    
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                
                reconstruction, sparse_hidden, _ = self.model(batch)
                
                all_sparse_hidden.append(sparse_hidden.cpu())
                all_reconstructions.append(reconstruction.cpu())
                all_targets.append(batch.cpu())
                
        # Concatenate results
        sparse_hidden = torch.cat(all_sparse_hidden, dim=0)
        reconstructions = torch.cat(all_reconstructions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute feature statistics
        feature_freq = (sparse_hidden > 0).float().mean(dim=0)
        feature_mean_activation = sparse_hidden.mean(dim=0)
        feature_max_activation = sparse_hidden.max(dim=0)[0]
        
        # Reconstruction quality
        mse = torch.mean((reconstructions - targets) ** 2, dim=0)
        cosine_sim = torch.nn.functional.cosine_similarity(
            targets, reconstructions, dim=1
        ).mean()
        
        analysis = {
            "feature_frequency": feature_freq,
            "feature_mean_activation": feature_mean_activation,
            "feature_max_activation": feature_max_activation,
            "mse_per_dim": mse,
            "overall_mse": mse.mean().item(),
            "cosine_similarity": cosine_sim.item(),
            "explained_variance": (
                1 - torch.var(targets - reconstructions) / torch.var(targets)
            ).item(),
            "num_dead_features": (feature_freq == 0).sum().item(),
            "sparsity": (sparse_hidden > 0).float().mean().item(),
        }
        
        return analysis 