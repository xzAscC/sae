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
        default="./scripts/trainer_{model_name}_Layer{model_layer}_{sae_name}SAE_{dataset}.yaml",
        help="Path to save the configuration",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-70m",
        help="LLM model name",
        choices=[
            "EleutherAI/pythia-70m",
            "google/gemma-2-2b",
            "Qwen/Qwen3-4B-Thinking-2507",
            "openai-community/gpt2",
        ],
    )
    parser.add_argument("--model_layer", type=int, default=3, help="LLM model layer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name",
        choices=["pyvene/axbench-concept16k_v2", "monology/pile-uncopyrighted"],
    )
    parser.add_argument(
        "--sae_name",
        type=str,
        default="topk",
        help="SAE name",
        choices=["topk", "absolutek"],
    )
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generator"
    )
    parser.add_argument(
        "--log_path", type=str, default="./logs", help="Path to save the logs"
    )
    parser.add_argument(
        "--dictionary_factor", type=int, default=8, help="Dictionary factor"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_ctxs", type=int, default=1000, help="Number of contexts")
    parser.add_argument("--ctx_len", type=int, default=1024, help="Context length")

    parser.add_argument("--steps", type=int, default=2000, help="Number of steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="save steps")
    parser.add_argument("--k", type=int, default=50, help="Number of top-k features")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--auxk_alpha", type=float, default=1/32, help="Auxiliary k")
    parser.add_argument("--decay_start", type=int, default=None, help="Decay start")
    parser.add_argument(
        "--threshold_beta", type=float, default=0.999, help="Threshold beta"
    )
    parser.add_argument(
        "--threshold_start_step", type=int, default=1000, help="Threshold start step"
    )

    # save the args
    def save_args(args: argparse.Namespace) -> None:
        config_file_path = args.config_save_path
        model_name = args.model_name.split("/")[-1]
        dataset_name = args.dataset.split("/")[-1]
        config_file_path = config_file_path.format(
            model_name=model_name,
            model_layer=args.model_layer,
            sae_name=args.sae_name,
            dataset=dataset_name,
        )
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)

            with open(config_file_path, "w") as f:
                yaml.dump(vars(args), f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file_path}: {e}")

    args = parser.parse_args()
    save_args(args)
    return args


def train_sae() -> None:
    # config setup
    args = config()

    # logger setup
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    logger.add(
        f"{args.log_path}/trainer_{model_name}_Layer{args.model_layer}_{args.sae_name}SAE_{dataset_name}.log",
        rotation="100 MB",
        retention="10 days",
    )
    logger.info(f"Training SAE with configuration: {args}")

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
    model_kwargs = {
        "device_map": device,
    }
    if args.model_name == "EleutherAI/pythia-70m":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model = nnsight.LanguageModel(args.model_name, **model_kwargs)
        model_layer_name = "gpt_neox"
        activation_dim = getattr(model, model_layer_name).embed_in.weight.shape[1]
        dictionary_dim = args.dictionary_factor * activation_dim
        logger.info(
            f"Activation dim: {activation_dim}, Dictionary dim: {dictionary_dim}"
        )
    elif args.model_name == "google/gemma-2-2b":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model = nnsight.LanguageModel(args.model_name, **model_kwargs)
        model_layer_name = "model"
        activation_dim = getattr(model, model_layer_name).embed_tokens.weight.shape[1]
        dictionary_dim = args.dictionary_factor * activation_dim
        logger.info(
            f"Activation dim: {activation_dim}, Dictionary dim: {dictionary_dim}"
        )
    elif args.model_name == "Qwen/Qwen3-4B-Thinking-2507":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model = nnsight.LanguageModel(args.model_name, **model_kwargs)
        model_layer_name = "model"
        activation_dim = getattr(model, model_layer_name).embed_tokens.weight.shape[1]
        dictionary_dim = args.dictionary_factor * activation_dim
        logger.info(
            f"Activation dim: {activation_dim}, Dictionary dim: {dictionary_dim}"
        )
    elif args.model_name == "openai-community/gpt2":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model = nnsight.LanguageModel(args.model_name, **model_kwargs)
        model_layer_name = "transformer"
        activation_dim = getattr(model, model_layer_name).wte.weight.shape[1]
        dictionary_dim = args.dictionary_factor * activation_dim
        logger.info(
            f"Activation dim: {activation_dim}, Dictionary dim: {dictionary_dim}"
        )
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    try:
        data = load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        data = iter(
            [
                "This is some example data",
                "In real life, for training a dictionary",
                "you would need much more data than this",
            ]
        )

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
    )  # buffer will yield batches of tensors of dimension = submodule's output dimension

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
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving SAE to: {save_dir}")

    # Calculate save steps for intermediate checkpoints (every 10% of total steps)
    save_steps = list(range(0, args.steps, args.save_steps))
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
            decay_start=args.decay_start,
            name=f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name}_{timestamp}",
        )
    else:
        raise ValueError(f"SAE {args.sae_name} not supported")

    notify_gmail(
        message=f"SAE{args.sae_name}_{model_name}_Layer{args.model_layer}_{dataset_name} training completed",
        subject=f"SAE Training Completed - {model_name} Layer {args.model_layer} {args.sae_name} SAE {args.dataset}"
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
