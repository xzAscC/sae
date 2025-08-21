"""
MSE vs Sparsity Trade-off Analysis for SAELens and local SAE.

This module analyzes the trade-off between Mean Squared Error (MSE) and sparsity
in Sparse AutoEncoders (SAE) using different sparsity regularization strengths.
"""

# TODO: clean this file
import sae_lens
import argparse
import sys
import sae_lens
import torch
import datasets
import nnsight
from loguru import logger
from sae import TopKSAE
import matplotlib.pyplot as plt

def config_setup():
    parser = argparse.ArgumentParser(
        description="Sparse AutoEncoder with MSE vs Sparsity Trade-off Analysis"
    )

    parser.add_argument(
        "--use_saelens", action="store_true", help="Use SAELens SAE", default=True
    )
    parser.add_argument(
        "--saelens_model_name",
        type=str,
        help="SAELens model name",
        default="sae_bench_gemma-2-2b_topk_width-2pow12_date-1109",
    )
    parser.add_argument(
        "--saelens_model_layer", type=int, help="SAELens model layer", default=12
    )

    return parser.parse_args()


def logger_setup():
    logger.add("logs/mse_sparsity_tradeoff.log", rotation="100 MB", retention="10 days")
    logger.add(sys.stdout, level="INFO")
    return logger

def load_sae_from_saelens(model_name: str, model_layer: int, trainer: int=5):
    if model_name == "sae_bench_gemma-2-2b_topk_width-2pow12_date-1109":
        sae = TopKSAE(activation_dim=2304, dict_size=4096, k=10)
        sae = sae.from_pretrained(use_saelens=True, model_name=model_name, model_layer=model_layer, trainer=trainer, device="cpu")
    elif model_name == "sae_bench_pythia70m_sweep_gated_ctx128_0730":
        sae = TopKSAE(activation_dim=2304, dict_size=4096, k=10)
        sae = sae.from_pretrained(use_saelens=True, model_name=model_name, model_layer=model_layer, trainer=trainer, device="cpu")
    else:
        raise ValueError(f"Model name {model_name} not supported")
    return sae

def plot_mse_sparsity_tradeoff() -> None:
    # Configuration
    args = config_setup()
    logger = logger_setup()
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 80)
    logger.info(f"args: {args}")
    if args.use_saelens:
        sae = load_sae_from_saelens(args.saelens_model_name, args.saelens_model_layer)
        sae.to(device)
    else:
        # TODO: load local SAE
        pass

    # Run analysis
    logger.info("Starting MSE vs Sparsity trade-off analysis...")
    data = datasets.load_dataset("pyvene/axbench-concept16k_v2", split="train")
    dataset = [data[i]["output"] for i in range(2)]
    model = nnsight.LanguageModel("google/gemma-2-2b", device_map="auto")
    model.to(torch.bfloat16)
    k_list = []
    mse_loss_list = []
    for k in range(10, 1000, 10):
        with model.trace(dataset):
            # For Gemma2 models, layers are accessed through model.model.layers
            res_x = model.model.layers[12].output.save()
        torch.cuda.empty_cache()
        res_x = res_x[0].reshape(-1, 2304).to(device)
        sae.k = k
        res_x_hat = sae.forward(res_x)
        mse_loss = torch.nn.functional.mse_loss(res_x_hat, res_x)
        logger.info(f"MSE loss: {mse_loss}, k: {k}")
        k_list.append(k)
        mse_loss_list.append(mse_loss.item())
        
    plt.plot(k_list, mse_loss_list, label="AbsoluteK")
    
    # Add red vertical line at k = 230
    plt.axvline(x=230, color='red', linestyle='--', linewidth=2, label='k = 230')
    
    plt.xlabel("k")
    plt.ylabel("MSE loss")
    plt.title("MSE vs Sparsity Trade-off")
    plt.legend()
    
    plt.savefig("mse_sparsity_tradeoff.png")
    plt.show()  # Display the plot
    
if __name__ == "__main__":
    plot_mse_sparsity_tradeoff() 
