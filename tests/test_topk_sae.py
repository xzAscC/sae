"""Tests for TopK SAE model."""

import pytest
import torch
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topk_sae.models.topk_sae import TopKSAE


class TestTopKSAE:
    """Test cases for TopK SAE model."""
    
    @pytest.fixture
    def sample_model(self) -> TopKSAE:
        """Create a sample TopK SAE model for testing."""
        return TopKSAE(
            input_dim=128,
            hidden_dim=512,
            k=32,
            tied_weights=True,
            normalize_decoder=True,
        )
        
    @pytest.fixture
    def sample_data(self) -> torch.Tensor:
        """Create sample activation data."""
        return torch.randn(16, 128)  # batch_size=16, input_dim=128
        
    def test_model_initialization(self) -> None:
        """Test model initialization with valid parameters."""
        model = TopKSAE(input_dim=64, hidden_dim=256, k=16)
        
        assert model.input_dim == 64
        assert model.hidden_dim == 256
        assert model.k == 16
        assert model.tied_weights == True  # default
        assert model.normalize_decoder == True  # default
        
    def test_model_initialization_invalid_params(self) -> None:
        """Test model initialization with invalid parameters."""
        # Invalid input_dim
        with pytest.raises(ValueError, match="input_dim must be positive"):
            TopKSAE(input_dim=0, hidden_dim=256, k=16)
            
        # Invalid hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            TopKSAE(input_dim=64, hidden_dim=0, k=16)
            
        # Invalid k (too large)
        with pytest.raises(ValueError, match="k must be in range"):
            TopKSAE(input_dim=64, hidden_dim=256, k=300)
            
        # Invalid k (zero)
        with pytest.raises(ValueError, match="k must be in range"):
            TopKSAE(input_dim=64, hidden_dim=256, k=0)
            
    def test_forward_pass(self, sample_model: TopKSAE, sample_data: torch.Tensor) -> None:
        """Test forward pass through the model."""
        reconstruction, sparse_hidden, loss_dict = sample_model(sample_data)
        
        # Check output shapes
        assert reconstruction.shape == sample_data.shape
        assert sparse_hidden.shape == (sample_data.shape[0], sample_model.hidden_dim)
        
        # Check sparsity
        num_active = (sparse_hidden > 0).sum(dim=-1)
        assert torch.all(num_active <= sample_model.k)
        
        # Check loss dictionary
        required_keys = [
            "reconstruction_loss", "l1_loss", "num_active", 
            "sparsity_ratio", "mean_activation", "max_activation"
        ]
        for key in required_keys:
            assert key in loss_dict
            assert isinstance(loss_dict[key], torch.Tensor)
            
    def test_encode_decode_cycle(self, sample_model: TopKSAE, sample_data: torch.Tensor) -> None:
        """Test encode -> apply_topk -> decode cycle."""
        # Encode
        hidden = sample_model.encode(sample_data)
        assert hidden.shape == (sample_data.shape[0], sample_model.hidden_dim)
        
        # Apply TopK
        sparse_hidden, topk_indices = sample_model.apply_topk(hidden)
        assert sparse_hidden.shape == hidden.shape
        assert topk_indices.shape == (sample_data.shape[0], sample_model.k)
        
        # Check sparsity constraint
        num_active = (sparse_hidden > 0).sum(dim=-1)
        assert torch.all(num_active == sample_model.k)
        
        # Decode
        reconstruction = sample_model.decode(sparse_hidden)
        assert reconstruction.shape == sample_data.shape
        
    def test_topk_sparsity(self, sample_model: TopKSAE) -> None:
        """Test that TopK sparsity is enforced correctly."""
        # Create test data
        hidden = torch.randn(8, sample_model.hidden_dim)
        
        # Apply TopK
        sparse_hidden, topk_indices = sample_model.apply_topk(hidden)
        
        # Check that exactly k values are non-zero per sample
        for i in range(hidden.shape[0]):
            non_zero_count = (sparse_hidden[i] > 0).sum().item()
            assert non_zero_count == sample_model.k
            
        # Check that TopK indices correspond to largest values
        for i in range(hidden.shape[0]):
            sample_hidden = hidden[i]
            sample_indices = topk_indices[i]
            
            # Get the k-th largest value
            kth_largest = torch.topk(sample_hidden, k=sample_model.k, largest=True)[0][-1]
            
            # All selected values should be >= k-th largest
            selected_values = sample_hidden[sample_indices]
            assert torch.all(selected_values >= kth_largest)
            
    def test_tied_weights(self) -> None:
        """Test tied weights functionality."""
        model_tied = TopKSAE(64, 256, 16, tied_weights=True)
        model_untied = TopKSAE(64, 256, 16, tied_weights=False)
        
        # Tied weights model should not have a separate decoder
        assert not hasattr(model_tied, 'decoder')
        assert hasattr(model_tied, 'decoder_bias') or model_tied.decoder_bias is None
        
        # Untied weights model should have a separate decoder
        assert hasattr(model_untied, 'decoder')
        
    def test_weight_normalization(self, sample_model: TopKSAE, sample_data: torch.Tensor) -> None:
        """Test decoder weight normalization."""
        sample_model.train()
        
        # Run forward pass (triggers normalization)
        _ = sample_model(sample_data)
        
        if sample_model.normalize_decoder:
            if sample_model.tied_weights:
                # Check encoder weights are normalized
                norms = torch.norm(sample_model.encoder.weight, dim=0)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
            else:
                # Check decoder weights are normalized
                norms = torch.norm(sample_model.decoder.weight, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
                
    def test_save_load_checkpoint(self, sample_model: TopKSAE) -> None:
        """Test model checkpointing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save checkpoint
            sample_model.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_model = TopKSAE.load_checkpoint(str(checkpoint_path))
            
            # Check model parameters match
            assert loaded_model.input_dim == sample_model.input_dim
            assert loaded_model.hidden_dim == sample_model.hidden_dim
            assert loaded_model.k == sample_model.k
            assert loaded_model.tied_weights == sample_model.tied_weights
            assert loaded_model.normalize_decoder == sample_model.normalize_decoder
            
            # Check state dict matches
            for key in sample_model.state_dict():
                assert torch.allclose(
                    sample_model.state_dict()[key],
                    loaded_model.state_dict()[key]
                )
                
    def test_reconstruction_metrics(self, sample_model: TopKSAE, sample_data: torch.Tensor) -> None:
        """Test reconstruction metrics computation."""
        metrics = sample_model.get_reconstruction_metrics(sample_data)
        
        required_metrics = [
            "mse", "cosine_similarity", "explained_variance", 
            "sparsity", "l1_norm"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (float, int))
            
        # Check metric ranges
        assert metrics["sparsity"] >= 0 and metrics["sparsity"] <= 1
        assert metrics["cosine_similarity"] >= -1 and metrics["cosine_similarity"] <= 1
        assert metrics["mse"] >= 0
        assert metrics["l1_norm"] >= 0
        
    def test_gradient_flow(self, sample_model: TopKSAE, sample_data: torch.Tensor) -> None:
        """Test that gradients flow through the model correctly."""
        sample_model.train()
        
        # Forward pass
        reconstruction, sparse_hidden, loss_dict = sample_model(sample_data)
        loss = loss_dict["reconstruction_loss"]
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for encoder
        assert sample_model.encoder.weight.grad is not None
        assert sample_model.encoder.weight.grad.norm() > 0
        
        if sample_model.encoder.bias is not None:
            assert sample_model.encoder.bias.grad is not None
            
        # Check decoder gradients
        if sample_model.tied_weights:
            if sample_model.decoder_bias is not None:
                assert sample_model.decoder_bias.grad is not None
        else:
            assert sample_model.decoder.weight.grad is not None
            if sample_model.decoder.bias is not None:
                assert sample_model.decoder.bias.grad is not None
                
    def test_different_k_values(self) -> None:
        """Test model with different k values."""
        input_dim, hidden_dim = 64, 256
        test_data = torch.randn(8, input_dim)
        
        for k in [1, 16, 32, 64, 128]:
            if k <= hidden_dim:
                model = TopKSAE(input_dim, hidden_dim, k)
                reconstruction, sparse_hidden, loss_dict = model(test_data)
                
                # Check sparsity
                actual_sparsity = (sparse_hidden > 0).float().mean()
                expected_sparsity = k / hidden_dim
                assert abs(actual_sparsity - expected_sparsity) < 0.01
                
    def test_batch_processing(self, sample_model: TopKSAE) -> None:
        """Test model with different batch sizes."""
        input_dim = sample_model.input_dim
        
        for batch_size in [1, 4, 16, 32]:
            test_data = torch.randn(batch_size, input_dim)
            reconstruction, sparse_hidden, loss_dict = sample_model(test_data)
            
            assert reconstruction.shape[0] == batch_size
            assert sparse_hidden.shape[0] == batch_size
            
            # Check that all samples have correct sparsity
            num_active = (sparse_hidden > 0).sum(dim=-1)
            assert torch.all(num_active == sample_model.k)


# Integration test (requires more setup)
@pytest.mark.slow
class TestTopKSAEIntegration:
    """Integration tests for TopK SAE."""
    
    def test_small_training_loop(self) -> None:
        """Test a small training loop end-to-end."""
        # Create small model and data
        model = TopKSAE(input_dim=32, hidden_dim=128, k=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create synthetic data
        train_data = torch.randn(100, 32)
        
        model.train()
        initial_loss = None
        final_loss = None
        
        # Training loop
        for epoch in range(5):
            epoch_loss = 0
            for i in range(0, len(train_data), 16):  # batch_size=16
                batch = train_data[i:i+16]
                
                # Forward pass
                reconstruction, sparse_hidden, loss_dict = model(batch)
                loss = loss_dict["reconstruction_loss"] + 1e-4 * loss_dict["l1_loss"]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / (len(train_data) // 16)
            
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss
            
        # Training should reduce loss
        assert final_loss < initial_loss
        print(f"Loss reduced from {initial_loss:.4f} to {final_loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__]) 