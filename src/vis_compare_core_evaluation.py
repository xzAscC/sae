import yaml
import safetensors
import transformer_lens
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import re
import glob
from data import ActivationsStore
from sae import BatchTopKSAE, JumpReLUSAE, BatchAbsoluteKSAE
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
        cfg = json.load(f)
    
    # Set device and dtype
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle dtype conversion - config may have string representation
    if "dtype" in cfg and isinstance(cfg["dtype"], str):
        if cfg["dtype"] == "torch.bfloat16":
            cfg["dtype"] = torch.bfloat16
        elif cfg["dtype"] == "torch.float32":
            cfg["dtype"] = torch.float32
        elif cfg["dtype"] == "torch.float16":
            cfg["dtype"] = torch.float16
        else:
            cfg["dtype"] = torch.float32
    else:
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


def generate_cache_key(cfg: dict, sae_type: str) -> str:
    """Generate a cache key based on model name, layer, and SAE type.
    
    Args:
        cfg: Configuration dictionary containing model_name and layer info
        sae_type: Type of SAE ('topk', 'absolutek', 'jumprelu')
    
    Returns:
        str: Cache key in format 'sae_type_model_layer'
    """
    model_name = cfg.get("model_name", "unknown").replace("/", "_").replace("-", "_")
    layer = cfg.get("layer", "unknown")
    hook_point = cfg.get("hook_point", "")
    
    # Extract layer number from hook_point if layer not directly available
    if layer == "unknown" and "blocks" in hook_point:
        layer_match = re.search(r'blocks\.(\d+)', hook_point)
        if layer_match:
            layer = layer_match.group(1)
    
    return f"{sae_type}_{model_name}_layer{layer}"


def evaluate_sae_performance(sae, cfg, k_values: list[int], num_samples: int = 50, cache_key: str = None):
    """Evaluate SAE performance across different k values.
    
    Args:
        sae: The SAE model
        cfg: Configuration dictionary
        k_values: List of k values to test
        num_samples: Number of samples to average over
        cache_key: Base cache key (will be extended with model/layer info)
    
    Returns:
        tuple: (k_list, mse_list, ce_list)
    """
    # Generate detailed cache key including model and layer info
    if cache_key:
        detailed_cache_key = generate_cache_key(cfg, cache_key)
        cache_file = f"assets/{detailed_cache_key}_performance.json"
        
        # Check if cached results exist
        if os.path.exists(cache_file):
            print(f"Loading cached results from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return cached_data['k_values'], cached_data['mse_values'], cached_data['ce_values']
    else:
        detailed_cache_key = None
        cache_file = None
    
    model = transformer_lens.HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    
    k_results = []
    mse_results = []
    ce_results = []
    
    for k in tqdm(k_values, desc=f"Evaluating k values for {detailed_cache_key or 'SAE'}"):
        sae.cfg["k"] = k
        total_mse = 0
        total_ce = 0
        
        for _ in range(num_samples):
            batch = activations_store.next_batch()
            batch_tokens = activations_store.get_batch_tokens()[:cfg["batch_size"] // cfg["seq_len"]]
            is_jumprelu = isinstance(sae, JumpReLUSAE)
            if is_jumprelu:
                x, x_reconstruct, sae_output = sae(batch, return_dict=False, select_topK=True)
            else:
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
    
    # Save results to cache with detailed filename
    if detailed_cache_key and cache_file:
        os.makedirs("assets", exist_ok=True)
        cache_data = {
            'k_values': k_results,
            'mse_values': mse_results,
            'ce_values': ce_results,
            'num_samples': num_samples,
            'model_name': cfg.get("model_name", "unknown"),
            'layer': cfg.get("layer", "unknown"),
            'hook_point': cfg.get("hook_point", "unknown"),
            'sae_type': cache_key
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Results cached to {cache_file}")
    
    return k_results, mse_results, ce_results


def find_cache_file_by_pattern(sae_type: str, assets_dir: str = "assets"):
    """Find cache files matching the SAE type pattern.
    
    Args:
        sae_type: SAE type ('topk', 'absolutek', 'jumprelu')
        assets_dir: Directory to search for cache files
    
    Returns:
        str: Path to the first matching cache file, or None if not found
    """
    pattern = f"{assets_dir}/{sae_type}_*_performance.json"
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]  # Return the first match
    return None


def load_cached_performance_data(cfg: dict = None, sae_type: str = None, k_values: list[int] = None, cache_file_path: str = None):
    """Load cached performance data from JSON file.
    
    Args:
        cfg: Configuration dictionary containing model and layer info (optional for auto-detection)
        sae_type: SAE type ('topk', 'absolutek', 'jumprelu')
        k_values: Expected k values (for validation)
        cache_file_path: Direct path to cache file (overrides other parameters)
    
    Returns:
        tuple: (k_list, mse_list, ce_list) or None if not found
    """
    if cache_file_path:
        cache_file = cache_file_path
    elif cfg is not None and sae_type is not None:
        cache_key = generate_cache_key(cfg, sae_type)
        cache_file = f"assets/{cache_key}_performance.json"
    elif sae_type is not None:
        # Try to find cache file by pattern matching
        cache_file = find_cache_file_by_pattern(sae_type)
        if not cache_file:
            print(f"Warning: No cache file found for {sae_type} SAE!")
            return None, None, None
    else:
        raise ValueError("Must provide either cfg+sae_type or cache_file_path or sae_type for pattern matching")
    
    if os.path.exists(cache_file):
        print(f"Loading cached performance data from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            return cached_data['k_values'], cached_data['mse_values'], cached_data['ce_values']
    else:
        print(f"Warning: Cache file {cache_file} not found!")
        return None, None, None


def compare_three_sae_models(topk_cfg_path: str = None, topk_tensor_path: str = None, 
                            absolutek_cfg_path: str = None, absolutek_tensor_path: str = None,
                            jumprelu_cfg_path: str = None, jumprelu_tensor_path: str = None,
                            topk_loss_path: str = "assets/topk_pythia_3.csv",
                            absolutek_loss_path: str = "assets/absolutek_pythia_3.csv",
                            jumprelu_loss_path: str = "assets/jumprelu_pythia_3.csv",
                            k_values: list[int] = [10, 20, 30, 40, 50],
                            save_path: str = "sae_comparison.pdf",
                            preload_only: bool = False):
    """Compare TopK, AbsoluteK and JumpReLU SAE models across different k values.
    
    Args:
        topk_cfg_path: Path to TopK SAE config file (optional if preload_only=True)
        topk_tensor_path: Path to TopK SAE weights file (optional if preload_only=True)
        absolutek_cfg_path: Path to AbsoluteK SAE config file (optional if preload_only=True)
        absolutek_tensor_path: Path to AbsoluteK SAE weights file (optional if preload_only=True)
        jumprelu_cfg_path: Path to JumpReLU SAE config file (optional if preload_only=True)
        jumprelu_tensor_path: Path to JumpReLU SAE weights file (optional if preload_only=True)
        topk_loss_path: Path to TopK loss CSV file
        absolutek_loss_path: Path to AbsoluteK loss CSV file
        jumprelu_loss_path: Path to JumpReLU loss CSV file
        k_values: List of k values to test
        save_path: Path to save the comparison plot
        preload_only: If True, only load cached JSON data without running experiments
    """
    
    if preload_only:
        print("🚀 Preload mode: Loading cached experimental data only...")
        
        # Try to load cached performance data using pattern matching first
        print("Loading TopK SAE performance data...")
        topk_k, topk_mse, topk_ce = load_cached_performance_data(sae_type="topk", k_values=k_values)
        
        print("Loading AbsoluteK SAE performance data...")
        abs_k, abs_mse, abs_ce = load_cached_performance_data(sae_type="absolutek", k_values=k_values)
        
        print("Loading JumpReLU SAE performance data...")
        jumprelu_k, jumprelu_mse, jumprelu_ce = load_cached_performance_data(sae_type="jumprelu", k_values=k_values)
        
        # If pattern matching failed and we have config files, try using them
        if any(data is None for data in [topk_k, topk_mse, topk_ce, abs_k, abs_mse, abs_ce, jumprelu_k, jumprelu_mse, jumprelu_ce]):
            if any([topk_cfg_path, absolutek_cfg_path, jumprelu_cfg_path]):
                print("Pattern matching failed, trying with config files...")
                sample_cfg_path = topk_cfg_path or absolutek_cfg_path or jumprelu_cfg_path
                with open(sample_cfg_path, "r") as f:
                    sample_cfg = json.load(f)
                
                if topk_k is None:
                    topk_k, topk_mse, topk_ce = load_cached_performance_data(sample_cfg, "topk", k_values)
                if abs_k is None:
                    abs_k, abs_mse, abs_ce = load_cached_performance_data(sample_cfg, "absolutek", k_values)
                if jumprelu_k is None:
                    jumprelu_k, jumprelu_mse, jumprelu_ce = load_cached_performance_data(sample_cfg, "jumprelu", k_values)
        
        # Check if all data was loaded successfully
        if any(data is None for data in [topk_k, topk_mse, topk_ce, abs_k, abs_mse, abs_ce, jumprelu_k, jumprelu_mse, jumprelu_ce]):
            print("❌ Error: Some cached data is missing. Please run experiments first or set preload_only=False.")
            print("Available cache files:")
            cache_files = glob.glob("assets/*_performance.json")
            for cache_file in cache_files:
                print(f"  - {cache_file}")
            return None
            
    else:
        print("🔬 Full mode: Loading models and running experiments...")
        
        print("Loading TopK SAE...")
        topk_sae, topk_cfg = load_sae_model(topk_cfg_path, topk_tensor_path, "batchtopk")
        
        print("Loading AbsoluteK SAE...")
        abs_sae, abs_cfg = load_sae_model(absolutek_cfg_path, absolutek_tensor_path, "batchabsolutek")
        
        print("Loading JumpReLU SAE...")
        jumprelu_sae, jumprelu_cfg = load_sae_model(jumprelu_cfg_path, jumprelu_tensor_path, "jumprelu")
        
        print("Evaluating TopK SAE...")
        topk_k, topk_mse, topk_ce = evaluate_sae_performance(topk_sae, topk_cfg, k_values, cache_key="topk")
        
        print("Evaluating AbsoluteK SAE...")
        abs_k, abs_mse, abs_ce = evaluate_sae_performance(abs_sae, abs_cfg, k_values, cache_key="absolutek")
        
        print("Evaluating JumpReLU SAE...")
        jumprelu_k, jumprelu_mse, jumprelu_ce = evaluate_sae_performance(jumprelu_sae, jumprelu_cfg, k_values, cache_key="jumprelu")
    
    # Load training loss data
    print("Loading training loss data...")
    topk_loss_df = pd.read_csv(topk_loss_path)
    abs_loss_df = pd.read_csv(absolutek_loss_path)
    
    # Try to load JumpReLU loss data if it exists
    jumprelu_loss_df = None
    if os.path.exists(jumprelu_loss_path):
        jumprelu_loss_df = pd.read_csv(jumprelu_loss_path)
        jumprelu_loss_df["value"] = jumprelu_loss_df["value"] - 0.09454367983341217
    
    # Create horizontal comparison plots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Training Loss Comparison
    ax1.plot(topk_loss_df["step"], topk_loss_df["value"], color="tab:blue", label="TopK SAE", linewidth=2)
    ax1.plot(abs_loss_df["step"], abs_loss_df["value"], color="tab:red", label="AbsoluteK SAE", linewidth=2)
    if jumprelu_loss_df is not None:
        ax1.plot(jumprelu_loss_df["step"], jumprelu_loss_df["value"], color="tab:green", label="JumpReLU SAE", linewidth=2)
    ax1.set_xlabel("Training Steps", fontsize=20, fontweight="bold")
    ax1.set_ylabel("MSE Loss", fontsize=20, fontweight="bold")
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
    ax1.set_yscale("log")
    ax1.tick_params(axis='both', which='both', labelsize=18)
    
    # Plot 2: MSE Comparison
    ax2.plot(topk_k, topk_mse, marker="o", color="tab:blue", label="TopK SAE", linewidth=2, markersize=8)
    ax2.plot(abs_k, abs_mse, marker="s", color="tab:red", label="AbsoluteK SAE", linewidth=2, markersize=8)
    ax2.plot(jumprelu_k, jumprelu_mse, marker="^", color="tab:green", label="JumpReLU SAE", linewidth=2, markersize=8)
    ax2.set_xlabel("number of active features", fontsize=20, fontweight="bold")
    ax2.set_ylabel("Normalized MSE", fontsize=20, fontweight="bold")
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
    ax2.set_yscale("log")
    ax2.tick_params(axis='both', which='both', labelsize=18)
    
    # Plot 3: CE Comparison
    ax3.plot(topk_k, topk_ce, marker="o", color="tab:blue", label="TopK SAE", linewidth=2, markersize=8)
    ax3.plot(abs_k, abs_ce, marker="s", color="tab:red", label="AbsoluteK SAE", linewidth=2, markersize=8)
    ax3.plot(jumprelu_k, jumprelu_ce, marker="^", color="tab:green", label="JumpReLU SAE", linewidth=2, markersize=8)
    ax3.set_xlabel("number of active features", fontsize=20, fontweight="bold")
    ax3.set_ylabel("CE Loss Score", fontsize=20, fontweight="bold")
    ax3.legend(fontsize=16)
    ax3.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
    ax3.tick_params(axis='both', which='both', labelsize=18)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print summary
    print(f"\nComparison Results:")
    print(f"TopK SAE - Final Loss: {topk_loss_df['value'].iloc[-1]:.6f}, MSE: {np.mean(topk_mse):.6f}, CE: {np.mean(topk_ce):.6f}")
    print(f"AbsoluteK SAE - Final Loss: {abs_loss_df['value'].iloc[-1]:.6f}, MSE: {np.mean(abs_mse):.6f}, CE: {np.mean(abs_ce):.6f}")
    print(f"JumpReLU SAE - MSE: {np.mean(jumprelu_mse):.6f}, CE: {np.mean(jumprelu_ce):.6f}")
    
    return {
        "topk": {"k": topk_k, "mse": topk_mse, "ce": topk_ce, "loss": topk_loss_df},
        "absolutek": {"k": abs_k, "mse": abs_mse, "ce": abs_ce, "loss": abs_loss_df},
        "jumprelu": {"k": jumprelu_k, "mse": jumprelu_mse, "ce": jumprelu_ce, "loss": jumprelu_loss_df}
    }


if __name__ == "__main__":

    
    results_preload = compare_three_sae_models(
        k_values=[10, 20, 30, 40, 50],
        save_path="assets/sae_comparison_three_models_pythia70m_layer3.pdf",
        preload_only=True  
    )
    
    if results_preload is None:
        print("\n" + "=" * 60)
        print("🔬 FULL MODE: Running experiments and caching results")
        print("=" * 60)
        
        topk_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/config_25000.json"
        topk_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/sae_10000.safetensors"
        
        absolutek_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchabsolutek_51_0.0003/config_25000.json"
        absolutek_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchabsolutek_51_0.0003/sae_30000.safetensors"
        
        jumprelu_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_jumprelu_51_0.0003/config_25000.json"
        jumprelu_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_jumprelu_51_0.0003/sae_10000.safetensors"
        
        results_full = compare_three_sae_models(
            topk_cfg_path=topk_cfg_path,
            topk_tensor_path=topk_tensor_path,
            absolutek_cfg_path=absolutek_cfg_path,
            absolutek_tensor_path=absolutek_tensor_path,
            jumprelu_cfg_path=jumprelu_cfg_path,
            jumprelu_tensor_path=jumprelu_tensor_path,
            k_values=[10, 20, 30, 40, 50],
            save_path="assets/sae_comparison_three_models_pythia70m_layer3.pdf",
            preload_only=False 
        )
    print("\n✅ Comparison completed! Check the generated PDF files.")