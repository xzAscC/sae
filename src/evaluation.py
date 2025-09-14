import torch
import nnsight
import argparse
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Optional
from loguru import logger
from sae import AbsoluteKSAE
from data import load_dataset, ActivationBuffer


def _to_input_ids(
    data: Union[str, List[str], torch.Tensor], model: nnsight.LanguageModel, device: str
) -> torch.Tensor:
    """
    Ensure input is token ids tensor [batch, seq].
    """
    if isinstance(data, torch.Tensor):
        input_ids = data
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids.to(device)
    if isinstance(data, str):
        enc = model.tokenizer(
            data, return_tensors="pt", truncation=True, max_length=1024
        )
        return enc.input_ids.to(device)
    if isinstance(data, list) and all(isinstance(s, str) for s in data):
        enc = model.tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        return enc.input_ids.to(device)
    raise ValueError("Unsupported data type for tokenization")


def _next_token_ce(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute next-token cross-entropy given logits [B, T, V] and labels from input_ids [B, T].
    """
    # Shift for next-token prediction
    logits_shifted = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    B, Tm1, V = logits_shifted.shape
    
    # Ensure labels are within valid vocabulary range and handle padding
    vocab_size = logits_shifted.size(-1)
    labels = labels.clamp(0, vocab_size - 1)  # Clamp to valid range
    
    # Create mask for valid tokens (non-padding)
    if hasattr(input_ids, 'attention_mask'):
        # If we have attention mask, use it
        mask = input_ids.attention_mask[:, 1:].contiguous()
    else:
        # Assume padding token is 0 or very large values
        mask = (labels >= 0) & (labels < vocab_size)
    
    # Compute loss only on valid tokens
    loss = F.cross_entropy(
        logits_shifted.view(-1, V), 
        labels.view(-1), 
        reduction="none"
    )
    
    # Apply mask and compute mean
    loss = loss.view(B, Tm1)
    if mask is not None:
        mask = mask.float()
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    
    return loss


def explained_variance(
    sae: torch.nn.Module,
    original_activations: torch.Tensor,
    reconstructed_activations: torch.Tensor,
) -> float:
    """
    Compute explained variance ratio.

    Args:
        sae: The SAE model
        original_activations: Original model activations
        reconstructed_activations: SAE reconstructed activations

    Returns:
        Explained variance ratio
    """
    with torch.no_grad():
        # Compute variance explained
        total_variance = torch.var(original_activations, dim=0).sum()
        residual_variance = torch.var(
            original_activations - reconstructed_activations, dim=0
        ).sum()

        explained_variance_ratio = 1 - (residual_variance / total_variance)

    logger.info(f"Explained Variance Ratio: {explained_variance_ratio.item():.6f}")
    return explained_variance_ratio.item()


def feature_density_statistics(sae: torch.nn.Module, activations: torch.Tensor) -> dict:
    """
    Compute feature density statistics for SAE.

    Args:
        sae: The SAE model
        activations: Input activations

    Returns:
        Dictionary with feature statistics
    """
    sae.eval()

    with torch.no_grad():
        # Get SAE features
        features = sae.encode(activations)

        # Compute statistics
        active_features = (features > 0).float()
        feature_density = active_features.mean(dim=0)  # Per-feature activation rate

        stats = {
            "mean_density": float(feature_density.mean()),
            "std_density": float(feature_density.std()),
            "min_density": float(feature_density.min()),
            "max_density": float(feature_density.max()),
            "dead_features": int((feature_density == 0).sum()),
            "total_features": len(feature_density),
            "sparsity": float(1 - feature_density.mean()),
        }

    logger.info(f"Feature Density Statistics: {stats}")
    return stats


def comprehensive_evaluation(
    sae: torch.nn.Module,
    model: nnsight.LanguageModel,
    data_loader,
    model_layer: int,
    model_name: str = "EleutherAI/pythia-70m",
    steps: int = 10,
    device: str = "cuda",
    activation_dim: int = 1024,
) -> dict:
    """
    Run comprehensive evaluation of SAE.

    Args:
        sae: The trained SAE model
        model: The language model to evaluate
        data_loader: DataLoader with evaluation data
        model_layer: Which layer to inject SAE reconstruction
        model_name: Name of the model
        device: Device to run on

    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("Starting comprehensive SAE evaluation...")
    kl_scores = []
    nce_scores = []
    for batch_idx, data in enumerate(data_loader):
        if batch_idx > steps:
            break
        with model.trace(data, invoker_args={"truncation": True, "max_length": 1024}):
            # SAE reconstruction
            if "pythia" in model_name.lower():
                x = model.gpt_neox.layers[model_layer].output.save()
                original_logits = model.gpt_neox.output.save()
            elif "gpt2" in model_name.lower():
                x = model.transformer.h[model_layer].output.save()
                original_logits = model.transformer.output.save()
            elif "gemma" in model_name.lower():
                x = model.model.layers[model_layer].output.save()
                original_logits = model.model.output.save()
            elif "qwen" in model_name.lower():
                x = model.model.layers[model_layer].output.save()
                original_logits = model.model.output.save()
            x_hat = sae.decode(sae.encode(x[0]))
            if "pythia" in model_name.lower():
                model.gpt_neox.layers[model_layer].output = x_hat
                new_logits = model.gpt_neox.output.save()
            elif "gpt2" in model_name.lower():
                model.transformer.h[model_layer].output = x_hat
                new_logits = model.transformer.output.save()
            elif "gemma" in model_name.lower():
                model.model.layers[model_layer].output = x_hat
                new_logits = model.model.output.save()
            elif "qwen" in model_name.lower():
                model.model.layers[model_layer].output = x_hat
                new_logits = model.model.output.save()
                
        # original ablation
        with model.trace(data, invoker_args={"truncation": True, "max_length": 1024}):
            if "pythia" in model_name.lower():
                original_logits2 = model.gpt_neox.output.save()
            elif "gpt2" in model_name.lower():
                original_logits2 = model.transformer.output.save()
            elif "gemma" in model_name.lower():
                original_logits2 = model.model.output.save()
            elif "qwen" in model_name.lower():
                original_logits2 = model.model.output.save()
        
        # zero ablation
        with model.trace(data, invoker_args={"truncation": True, "max_length": 1024}):

            # zero ablation
            if "pythia" in model_name.lower():
                model.gpt_neox.layers[model_layer].output[0].zero_()
                new_logits_zero = model.gpt_neox.output.save()
            elif "gpt2" in model_name.lower():
                model.transformer.h[model_layer].output[0].zero_()
                new_logits_zero = model.transformer.output.save()
            elif "gemma" in model_name.lower():
                model.model.layers[model_layer].output[0].zero_()
                new_logits_zero = model.model.output.save()
            elif "qwen" in model_name.lower():
                model.model.layers[model_layer].output[0].zero_()
                new_logits_zero = model.model.output.save()

        # Compute KL divergence
        kl_div_score = F.kl_div(
            F.log_softmax(new_logits[0], dim=-1),
            F.softmax(original_logits2[0], dim=-1),  
            reduction="batchmean",
        )
        kl_scores.append(kl_div_score.item())

        # Compute normalized CE metric
        input_ids = _to_input_ids(data, model, device)
        H_orig = _next_token_ce(original_logits2[0], input_ids)
        H_star = _next_token_ce(new_logits[0], input_ids)
        H_0 = _next_token_ce(new_logits_zero[0], input_ids)
        denom = (H_orig - H_0)
        # TODO: probably divide by zero
        metric = (H_star - H_0) / denom
        value = float(metric.item())
        nce_scores.append(value)

    np_kl_scores = np.array(kl_scores)
    logger.info(
        f"KL Divergence Score: {np_kl_scores.mean():.6f} ± {np_kl_scores.std():.6f}"
    )
    np_nce_scores = np.array(nce_scores)
    logger.info(
        f"Normalized CE metric: {np_nce_scores.mean():.6f} ± {np_nce_scores.std():.6f}"
    )
    logger.info("Comprehensive evaluation completed.")
    return None


def evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_layer", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", choices=["EleutherAI/pythia-70m", "google/gemma-2-2b", "Qwen/Qwen3-4B-Thinking-2507", "openai-community/gpt2"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sae_path",
        type=str,
        default="logs/SAEabsolutek_gemma-2-2b_Layer12_pile-uncopyrighted_20250914_014701/checkpoints/SAE_checkpoint_step_0.safetensors",
    )
    parser.add_argument("--data_path", type=str, default="pyvene/axbench-concept16k_v2")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--dictionary_factor", type=int, default=8)
    parser.add_argument("--n_ctxs", type=int, default=1000)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--remove_bos", type=bool, default=False)
    parser.add_argument("--add_special_token", type=bool, default=True)
    parser.add_argument("--k", type=int, default=230)
    args = parser.parse_args()

    torch_dtype = getattr(torch, args.torch_dtype)
    device = args.device

    if args.model_name == "EleutherAI/pythia-70m":
        model = nnsight.LanguageModel(
            args.model_name, device_map=device, torch_dtype=torch_dtype
        )
        model_layer_name = "gpt_neox"
        activation_dim = getattr(model, model_layer_name).embed_in.weight.shape[1]

    elif args.model_name == "google/gemma-2-2b":
        model = nnsight.LanguageModel(
            args.model_name, device_map=device, torch_dtype=torch_dtype
        )
        model_layer_name = "model"
        activation_dim = getattr(model, model_layer_name).embed_tokens.weight.shape[1]

    elif args.model_name == "Qwen/Qwen3-4B-Thinking-2507":
        model = nnsight.LanguageModel(
            args.model_name, device_map=device, torch_dtype=torch_dtype
        )
        model_layer_name = "model"
        activation_dim = getattr(model, model_layer_name).embed_tokens.weight.shape[1]
        dictionary_dim = args.dictionary_factor * activation_dim

    elif args.model_name == "openai-community/gpt2":
        model = nnsight.LanguageModel(
            args.model_name, device_map=device, torch_dtype=torch_dtype
        )
        model_layer_name = "transformer"
        activation_dim = getattr(model, model_layer_name).wte.weight.shape[1]

    else:
        raise ValueError(f"Model {args.model_name} not supported")

    dictionary_dim = args.dictionary_factor * activation_dim
    sae = AbsoluteKSAE(
        activation_dim=activation_dim,
        dict_size=dictionary_dim,
        k=args.k,
    )
    sae.from_pretrained(args.sae_path)
    sae.to(device).to(torch_dtype)
    data = load_dataset(args.data_path)

    evaluation_results = comprehensive_evaluation(
        sae,
        model,
        data,
        model_layer=args.model_layer,
        model_name=args.model_name,
        device=args.device,
        activation_dim=activation_dim,
    )
    logger.info(evaluation_results)


if __name__ == "__main__":
    evaluation()
