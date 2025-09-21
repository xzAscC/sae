import yaml
import safetensors
import transformer_lens
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import ActivationsStore
from sae import BatchTopKSAE
from functools import partial
from tqdm import tqdm


def normalize_MSE(reconstructed_activations, input_tensor):
    normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (
        input_tensor
    ).pow(2).sum(dim=1)
    return normalized_mse


def CE_degradation(original_loss, reconstructed_loss):
    return original_loss - reconstructed_loss


def vis_compare_core_evaluation(
    original_loss, reconstructed_loss, zero_loss, mean_loss
):
    return original_loss - reconstructed_loss


# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    # The sae_out should match the shape of the original activation
    # If shapes don't match, reshape sae_out to match activation shape
    if sae_out.shape != activation.shape:
        # Reshape sae_out to match activation shape
        return sae_out.view(activation.shape)
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def load_sae_model(cfg_path: str, tensor_path: str, sae_type: str):
    """Load an SAE model from config and checkpoint files.
    
    Args:
        cfg_path: Path to the configuration JSON file
        tensor_path: Path to the SAE weights file
        sae_type: Type of SAE ('batchtopk', 'batchabsolutek', 'jumprelu')
    
    Returns:
        tuple: (sae_model, config_dict)
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set device and dtype
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["dtype"] = torch.float32
    
    # Import the correct SAE class
    if sae_type == "batchtopk":
        from sae import BatchTopKSAE
        sae = BatchTopKSAE(cfg)
    elif sae_type == "batchabsolutek":
        from sae import BatchAbsoluteKSAE
        sae = BatchAbsoluteKSAE(cfg)
    elif sae_type == "jumprelu":
        from sae import JumpReLUSAE
        sae = JumpReLUSAE(cfg)
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
    
    sae.load_state_dict(safetensors.torch.load_file(tensor_path))
    return sae, cfg


def evaluate_sae_performance(sae, cfg, k_values: list[int], num_samples: int = 50):
    """Evaluate SAE performance across different k values.
    
    Args:
        sae: The SAE model
        cfg: Configuration dictionary
        k_values: List of k values to test
        num_samples: Number of samples to average over
    
    Returns:
        tuple: (k_list, mse_list, ce_list)
    """
    model = transformer_lens.HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    
    k_results = []
    mse_results = []
    ce_results = []
    
    for k in tqdm(k_values, desc="Evaluating k values"):
        sae.cfg["k"] = k
        total_mse = 0
        total_ce = 0
        
        for _ in range(num_samples):
            batch = activations_store.next_batch()
            batch_tokens = activations_store.get_batch_tokens()[:cfg["batch_size"] // cfg["seq_len"]]
            
            # Get SAE reconstruction
            x, x_reconstruct, sae_output = sae(batch, return_dict=False)
            normalized_mse = normalize_MSE(x_reconstruct, x)
            
            # The SAE output has shape (batch_size * seq_len, act_size)
            # We need to use it directly as the hook replacement
            # The hook expects the same shape as the original activation
            sae_output_for_hook = sae_output["sae_out"]
            
            # Calculate losses
            original_loss = model(batch_tokens, return_type="loss").item()
            
            reconstr_loss = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output_for_hook))],
                return_type="loss",
            ).item()
            
            zero_loss = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(cfg["hook_point"], zero_abl_hook)],
                return_type="loss",
            ).item()
            
            # Calculate metrics
            ce_degradation = original_loss - reconstr_loss
            zero_degradation = original_loss - zero_loss
            relative_ce = ce_degradation / zero_degradation if zero_degradation != 0 else 0
            
            total_mse += normalized_mse.mean().item()
            total_ce += relative_ce
        
        k_results.append(k)
        mse_results.append(total_mse / num_samples)
        ce_results.append(total_ce / num_samples)
    
    return k_results, mse_results, ce_results


def compare_sae_models(topk_cfg_path: str, topk_tensor_path: str, 
                      absolutek_cfg_path: str, absolutek_tensor_path: str,
                      topk_loss_path: str = "configs/asset/loss1.csv",
                      absolutek_loss_path: str = "configs/asset/loss2.csv",
                      k_values: list[int] = [10, 20, 30, 40, 50],
                      save_path: str = "sae_comparison.pdf"):
    """Compare TopK and AbsoluteK SAE models across different k values.
    
    Args:
        topk_cfg_path: Path to TopK SAE config file
        topk_tensor_path: Path to TopK SAE weights file
        absolutek_cfg_path: Path to AbsoluteK SAE config file
        absolutek_tensor_path: Path to AbsoluteK SAE weights file
        topk_loss_path: Path to TopK loss CSV file
        absolutek_loss_path: Path to AbsoluteK loss CSV file
        k_values: List of k values to test
        save_path: Path to save the comparison plot
    """
    print("Loading TopK SAE...")
    topk_sae, topk_cfg = load_sae_model(topk_cfg_path, topk_tensor_path, "batchtopk")
    
    print("Loading AbsoluteK SAE...")
    abs_sae, abs_cfg = load_sae_model(absolutek_cfg_path, absolutek_tensor_path, "batchabsolutek")
    
    print("Evaluating TopK SAE...")
    topk_k, topk_mse, topk_ce = evaluate_sae_performance(topk_sae, topk_cfg, k_values)
    
    print("Evaluating AbsoluteK SAE...")
    abs_k, abs_mse, abs_ce = evaluate_sae_performance(abs_sae, abs_cfg, k_values)
    
    # Load training loss data
    print("Loading training loss data...")
    topk_loss_df = pd.read_csv(topk_loss_path)
    abs_loss_df = pd.read_csv(absolutek_loss_path)
    
    # Create horizontal comparison plots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Training Loss Comparison
    ax1.plot(topk_loss_df["step"], topk_loss_df["value"], color="tab:blue", label="TopK SAE", linewidth=2)
    ax1.plot(abs_loss_df["step"], abs_loss_df["value"], color="tab:red", label="AbsoluteK SAE", linewidth=2)
    ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")
    
    # Plot 2: MSE Comparison
    ax2.plot(topk_k, topk_mse, marker="o", color="tab:blue", label="TopK SAE", linewidth=2, markersize=8)
    ax2.plot(abs_k, abs_mse, marker="s", color="tab:red", label="AbsoluteK SAE", linewidth=2, markersize=8)
    ax2.set_title("Normalized MSE", fontsize=14, fontweight="bold")
    ax2.set_xlabel("k (number of active features)", fontsize=12)
    ax2.set_ylabel("Normalized MSE", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
    
    # Plot 3: CE Comparison
    ax3.plot(topk_k, topk_ce, marker="o", color="tab:blue", label="TopK SAE", linewidth=2, markersize=8)
    ax3.plot(abs_k, abs_ce, marker="s", color="tab:red", label="AbsoluteK SAE", linewidth=2, markersize=8)
    ax3.set_title("Cross-Entropy Degradation", fontsize=14, fontweight="bold")
    ax3.set_xlabel("k (number of active features)", fontsize=12)
    ax3.set_ylabel("Relative CE Degradation", fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print summary
    print(f"\nComparison Results:")
    print(f"TopK SAE - Final Loss: {topk_loss_df['value'].iloc[-1]:.6f}, MSE: {np.mean(topk_mse):.6f}, CE: {np.mean(topk_ce):.6f}")
    print(f"AbsoluteK SAE - Final Loss: {abs_loss_df['value'].iloc[-1]:.6f}, MSE: {np.mean(abs_mse):.6f}, CE: {np.mean(abs_ce):.6f}")
    
    return {
        "topk": {"k": topk_k, "mse": topk_mse, "ce": topk_ce, "loss": topk_loss_df},
        "absolutek": {"k": abs_k, "mse": abs_mse, "ce": abs_ce, "loss": abs_loss_df}
    }


if __name__ == "__main__":
    # Define paths for Pythia-70M layer 3 SAEs
    topk_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/config_25000.json"
    topk_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/sae_5000.safetensors"
    
    absolutek_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchabsolutek_51_0.0003/config_25000.json"
    absolutek_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchabsolutek_51_0.0003/sae_30000.safetensors"
    
    # Run comparison
    results = compare_sae_models(
        topk_cfg_path=topk_cfg_path,
        topk_tensor_path=topk_tensor_path,
        absolutek_cfg_path=absolutek_cfg_path,
        absolutek_tensor_path=absolutek_tensor_path,
        k_values=[10, 20, 30, 40, 50],
        save_path="sae_comparison_pythia70m_layer3.pdf"
    )