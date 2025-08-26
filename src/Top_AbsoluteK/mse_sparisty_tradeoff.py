"""
    MSE vs Sparsity Trade-off Analysis for SAELens and local SAE.

This module analyzes the trade-off between Mean Squared Error (MSE) and sparsity
in Sparse AutoEncoders (SAE) using different sparsity regularization strengths.
"""

# TODO: clean this file
import argparse
import os
import sys
import torch
import datasets
import nnsight
from loguru import logger
from sae import TopKSAE, AbsoluteKSAE
import matplotlib.pyplot as plt


def config_setup():
    parser = argparse.ArgumentParser(
        description="Sparse AutoEncoder with MSE vs Sparsity Trade-off Analysis"
    )
    parser.add_argument(
        "--sae_name",
        type=str,
        help="SAE name",
        default="topk",
        choices=["topk", "absolutek"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        default="EleutherAI/pythia-70m",
        choices=[
            "google/gemma-2-2b",
            "EleutherAI/pythia-70m",
            "Qwen/Qwen3-4B-Thinking-2507",
            "openai-community/gpt2",
        ],
    )
    parser.add_argument("--model_layer", type=int, help="Model layer", default=3)
    parser.add_argument("--k", type=int, help="k", default=230)
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset",
        default="pyvene/axbench-concept16k_v2",
        choices=["monology/pile-uncopyrighted", "pyvene/axbench-concept16k_v2"],
    )
    parser.add_argument(
        "--log_path", type=str, help="Log path", default="logs/mse_sparsity_tradeoff_absolutek"
    )
    parser.add_argument(
        "--local_sae_path",
        type=str,
        help="Local SAE path",
        default="logs/SAEtopk_pythia-70m_Layer3_pile-uncopyrighted_20250822_025537/SAE_final.safetensors",
    )
    parser.add_argument(
        "--dictionary_factor", type=int, help="Dictionary factor", default=8
    )

    return parser.parse_args()


def plot_mse_sparsity_tradeoff() -> None:
    # Configuration
    args = config_setup()
    logger.add(
        os.path.join(args.log_path, "mse_sparsity_tradeoff.log"),
        rotation="100 MB",
        retention="10 days",
    )
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 80)
    logger.info(f"args: {args}")
    model = nnsight.LanguageModel(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    if args.model_name == "google/gemma-2-2b":
        activation_dim = model.model.embed_tokens.weight.shape[1]
    elif args.model_name == "EleutherAI/pythia-70m":
        activation_dim = model.gpt_neox.embed_in.weight.shape[1]
    elif args.model_name == "Qwen/Qwen3-4B-Thinking-2507":
        activation_dim = model.model.embed_tokens.weight.shape[1]
    elif args.model_name == "openai-community/gpt2":
        activation_dim = model.transformer.wte.weight.shape[1]
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    dictionary_size = args.dictionary_factor * activation_dim
    if args.sae_name == "topk":
        sae = TopKSAE(
            activation_dim=activation_dim, dict_size=dictionary_size, k=args.k
        )
    elif args.sae_name == "absolutek":
        sae = AbsoluteKSAE(
            activation_dim=activation_dim, dict_size=dictionary_size, k=args.k
        )
    else:
        raise ValueError(f"Invalid SAE name: {args.sae_name}")

    sae.from_pretrained(args.local_sae_path, device=device)
    sae.to(device)

    # Run analysis
    logger.info("Starting MSE vs Sparsity trade-off analysis...")
    if args.dataset == "monology/pile-uncopyrighted":
        data = datasets.load_dataset(args.dataset, split="train")
        dataset = [data[i]["text"] for i in range(1)]
    elif args.dataset == "pyvene/axbench-concept16k_v2":
        data = datasets.load_dataset(args.dataset, split="train")
        dataset = [data[i]["output"] for i in range(1)]
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    k_list = []
    mse_loss_list = []
    for k in range(10, 1000, 10):
        with model.trace("This is a test sentence", invoker_args={"truncation": True, "max_length": 1024}):
            res_x = model.gpt_neox.layers[args.model_layer].output.save()
        res_x = res_x[0].to(device).reshape(-1, activation_dim).detach()

        sae.k = 230
        with torch.no_grad():
            loss = 0
            loss = sae.test(res_x)
        k_list.append(k)
        mse_loss_list.append(loss.item())
        print(mse_loss_list)
        exit()
        logger.info(f"k: {k}, loss: {loss.item()}")

    plt.plot(k_list, mse_loss_list, label=args.sae_name)

    plt.axvline(
        x=args.k, color="red", linestyle="--", linewidth=2, label=f"k = {args.k}"
    )

    plt.xlabel("k")
    plt.ylabel("MSE loss")
    plt.title("MSE vs Sparsity Trade-off")
    plt.legend()

    plt.savefig("mse_sparsity_tradeoff.png")


if __name__ == "__main__":
    plot_mse_sparsity_tradeoff()
