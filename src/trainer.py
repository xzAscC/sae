import argparse
import os
import yaml
import torch
import nnsight
import apprise
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

from utils import seed_setup
from data import ActivationBuffer, load_dataset
from sae import TopKSAE, AbsoluteKSAE

load_dotenv()

APPRISE_GMAIL = os.getenv("APPRISE_GMAIL")
APPRISE_PWD = os.getenv("APPRISE_PWD")


def config() -> argparse.Namespace:
    """
    Config setup

    Returns:
        args: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--config_save_path",
        type=str,
        default="./configs/",
        help="Path to save the configuration",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="LLM model name",
        choices=[
            "EleutherAI/pythia-70m",
            "google/gemma-2-2b",
            "Qwen/Qwen3-4B-Thinking-2507",
            "openai-community/gpt2",
        ],
    )
    parser.add_argument("--model_layer", type=int, default=12, help="LLM model layer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name",
        choices=["pyvene/axbench-concept16k_v2", "monology/pile-uncopyrighted", "Salesforce/wikitext"],
    )
    parser.add_argument(
        "--sae_name",
        type=str,
        default="absolutek",
        help="SAE name",
        choices=["topk", "absolutek"],
    )
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generator"
    )
    parser.add_argument(
        "--log_path", type=str, default="./logs/", help="Path to save the logs"
    )
    parser.add_argument(
        "--dictionary_factor", type=int, default=8, help="Dictionary factor"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--n_ctxs", type=int, default=10, help="Number of contexts")
    parser.add_argument("--ctx_len", type=int, default=128, help="Context length")

    parser.add_argument("--steps", type=int, default=20000, help="Number of steps")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="save ratio")
    parser.add_argument("--k", type=int, default=230, help="Number of top-k features")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--auxk_alpha", type=float, default=1 / 32, help="Auxiliary k")
    parser.add_argument("--decay_start", type=int, default=None, help="Decay start")
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16", help="Torch dtype"
    )
    parser.add_argument(
        "--threshold_beta", type=float, default=0.999, help="Threshold beta"
    )
    parser.add_argument(
        "--threshold_start_step", type=int, default=1000, help="Threshold start step"
    )
    parser.add_argument(
        "--remove_bos", type=bool, default=False, help="Remove BOS token"
    )
    parser.add_argument(
        "--add_special_token", type=bool, default=True, help="Add special token"
    )
    parser.add_argument(    
        "--normalize_activations", type=bool, default=False, help="Normalize activations"
    )

    args = parser.parse_args()

    return args


def save_args(args: argparse.Namespace) -> None:
    config_file_path = args.config_save_path
    if not os.path.exists(config_file_path):
        os.makedirs(config_file_path, exist_ok=True)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    config_file_path = os.path.join(
        config_file_path, f"{model_name}_{dataset_name}_{args.sae_name}SAE.yaml"
    )
    log_file_path = os.path.join(
        args.log_path, f"{model_name}_{dataset_name}_{args.sae_name}SAE.log"
    )

    with open(config_file_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, indent=2)
    logger.add(log_file_path, rotation="100 MB", retention="10 days")
    logger.info(f"Training SAE with configuration: {args}")
    logger.info(f"Configuration saved to {config_file_path}")
    return model_name, dataset_name


def train_sae() -> None:
    # Step 1: Config and logger setup
    args = config()
    model_name, dataset_name = save_args(args)

    if args.verbose:
        logger.info("Verbose mode is enabled")
    else:
        logger.info("Verbose mode is disabled")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA is not available")
        raise ValueError("CUDA is not available")
    else:
        logger.info(f"Using device: {device}")

    seed_setup(args.seed)

    # Step 2: Model setup
    torch_dtype = getattr(torch, args.torch_dtype)
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
    (
        logger.info(
            f"Activation dim: {activation_dim}, Dictionary dim: {dictionary_dim}"
        )
        if args.verbose
        else None
    )
    logger.info(f"Model layer name: {model_layer_name}") if args.verbose else None

    # Step 3: Data setup
    try:
        data = load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")

    buffer = ActivationBuffer(
        data=data,
        model=model,
        model_layer_name=model_layer_name,
        model_layer=args.model_layer,
        d_submodule=activation_dim,
        n_ctxs=args.n_ctxs,
        ctx_len=args.ctx_len,
        device=device,
        batch_size=args.batch_size,
        remove_bos=args.remove_bos,
        add_special_token=args.add_special_token,
    )

    # Step 4: SAE setup
    warmup_steps = max(1, min(1000, args.steps // 10, args.steps - 1))
    # Ensure sparsity_warmup_steps is at least warmup_steps and less than total steps
    sparsity_warmup_steps = max(
        warmup_steps, min(2000, args.steps // 5, args.steps - 1)
    )

    # Create a timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.log_path,
        f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name}_{timestamp}",
    )

    logger.info(f"Saving SAE to: {save_dir}") if args.verbose else None

    # Calculate save steps for intermediate checkpoints (every 10% of total steps)
    save_steps = list(range(0, args.steps, int(args.save_ratio * args.steps)))
    if args.steps - 1 not in save_steps:
        save_steps.append(args.steps - 1)

    if args.sae_name == "topk":
        sae = TopKSAE(
            activation_dim=activation_dim,
            dict_size=dictionary_dim,
            k=args.k,
        )
        sae.to(device)
        sae.train(
            data=buffer,
            warmup_steps=warmup_steps,
            sparsity_warmup_steps=sparsity_warmup_steps,
            save_dir=save_dir,
            save_steps=save_steps,
            steps=args.steps,
            device=args.device,
            lr=args.lr,
            auxk_alpha=args.auxk_alpha,
            decay_start=args.decay_start,
            threshold_beta=args.threshold_beta,
            threshold_start_step=args.threshold_start_step,
            name=f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name}_{timestamp}",
        )
    elif args.sae_name == "absolutek":
        sae = AbsoluteKSAE(
            activation_dim=activation_dim,
            dict_size=dictionary_dim,
            k=args.k,
        )
        sae.to(device)
        sae.train(
            data=buffer,
            warmup_steps=warmup_steps,
            sparsity_warmup_steps=sparsity_warmup_steps,
            save_dir=save_dir,
            save_steps=save_steps,
            steps=args.steps,
            device=args.device,
            lr=args.lr,
            auxk_alpha=args.auxk_alpha,
            normalize_activations=args.normalize_activations,
            torch_dtype=torch_dtype,
            decay_start=args.decay_start,
            name=f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name}_{timestamp}",
        )
        sae.mse_sparsity_tradeoff(x=buffer, steps=args.steps)
    else:
        raise ValueError(f"SAE {args.sae_name} not supported")

    notify_gmail(
        message=f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name} training completed",
        subject=f"SAE Training Completed - {model_name} Layer {args.model_layer} {args.sae_name} SAE {args.dataset}",
    )
    return None


def notify_gmail(message: str, subject: str = "SAE Training Notification") -> None:
    """
    Notify via gmail

    Args:
        message (str): Message to send
        subject (str): Subject of the email

    Returns:
        None
    """
    notifier = apprise.Apprise()
    notifier.add(f"mailto://{APPRISE_GMAIL}:{APPRISE_PWD}@gmail.com")

    notifier.notify(body=message, title=subject)
    return None


if __name__ == "__main__":
    train_sae()
