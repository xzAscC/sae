"""
Date: 2025-09-04
Author: Xudong Zhu

Compute the difference between two SAE features, compute the cosine similarity between the difference and the original features, steer the model with the difference for toy concept.
"""

import torch
import argparse
import nnsight
import os
import sae_lens
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from loguru import logger

from utils import seed_setup


def config():
    """
    Config the model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--dataset_name", type=str, default="data_name")
    parser.add_argument("--log_dir", type=str, default="./logs/DiffSAE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_layer", type=int, default=19)
    return parser.parse_args()


def DiffSAE():
    """
    Compute the difference between two SAE features, compute the cosine similarity between the difference and the original features, steer the model with the difference for toy concept.
    """
    args = config()
    os.makedirs(args.log_dir, exist_ok=True)
    seed_setup(args.seed)

    logger.add(
        os.path.join(
            args.log_dir,
            f"Model{args.model_name.split('/')[-1]}_Dataset{args.dataset_name.split('/')[-1]}_Seed{args.seed}_Layer{args.model_layer}.log",
        ),
        level="INFO",
    )
    logger.info(f"Config: {args}")


    model = nnsight.LanguageModel(
        args.model_name, device_map=args.device, torch_dtype=torch.bfloat16
    )
    if args.model_name == "EleutherAI/pythia-70m":
        model_layer_name = "gpt_neox"
    elif args.model_name == "google/gemma-2-2b-it":
        model_layer_name = "model"
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    if args.model_name == "EleutherAI/pythia-70m":
        release = "pythia-70m-deduped-res-sm"
        sae_id = f"blocks.{args.model_layer}.hook_resid_post"
        sae = (
            sae_lens.SAE.from_pretrained(release, sae_id)[0]
            .to(args.device)
            .to(torch.bfloat16)
        )
    elif args.model_name == "google/gemma-2-2b-it":
        release = "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109"
        sae_id = f"blocks.{args.model_layer}.hook_resid_post__trainer_5"
        sae = (
            sae_lens.SAE.from_pretrained(release, sae_id)[0]
            .to(args.device)
            .to(torch.bfloat16)
        )
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    data = [
        "Long live the Man! Long live the Man",
        "Long live the Woman! Long live the Woman",
    ]
    with model.trace(data, invoker_args={"truncation": True, "max_length": 1024}):
        hidden_states = getattr(model, model_layer_name).layers[args.model_layer].output

        sae_acts_king = torch.matmul(hidden_states[0][0, -2, :].unsqueeze(0), sae.W_enc)
        sae_acts_queen = torch.matmul(
            hidden_states[0][1, -2, :].unsqueeze(0), sae.W_enc
        )

        # Save the activations for processing outside the trace
        sae_acts_king_saved = sae_acts_king.save()
        sae_acts_queen_saved = sae_acts_queen.save()

        # getattr(model, model_layer_name).layers[args.model_layer].output[0][
        #     :, -2, :
        # ] = (10000 * gender_feature + hidden_states[0][:, -2, :])
        # # Use embed_out instead of lm_head for Pythia-70m
        # patched_logits = model.lm_head.output.save()

    # Now process the saved activations outside the trace context
    king_unique, queen_unique, min_king_unique, min_queen_unique = select_the_different_index(sae_acts_king_saved, sae_acts_queen_saved)
    
    data2 = [
        "Long live the King! Long live the King",
        "Long live the Queen! Long live the Queen",
        "Long live the Man! Long live the Man",
        "Long live the Woman! Long live the Woman",
    ]
    with model.trace(data2, invoker_args={"truncation": True, "max_length": 1024}):
        cs_list = torch.zeros(24)

        for layer in range(24):
            hidden_states = getattr(model, model_layer_name).layers[layer].output.save()
            MW_representation = hidden_states[0][0, -1, :] - hidden_states[0][1, -1, :]
            KQ_representation = hidden_states[0][2, -1, :] - hidden_states[0][3, -1, :]
            cs_list[layer] = torch.nn.functional.cosine_similarity(MW_representation.unsqueeze(0), KQ_representation.unsqueeze(0)).item()
        cs_list = cs_list.save()
        hidden_states = getattr(model, model_layer_name).layers[args.model_layer].output.save()

        sae_acts_king = torch.matmul(hidden_states[0][0, -2, :].unsqueeze(0), sae.W_enc)
        sae_acts_queen = torch.matmul(
            hidden_states[0][1, -2, :].unsqueeze(0), sae.W_enc
        )

        # Save the activations for processing outside the trace
        sae_acts_king_saved = sae_acts_king.save()
        sae_acts_queen_saved = sae_acts_queen.save()
    king_unique_2, queen_unique_2, min_king_unique_2, min_queen_unique_2 = select_the_different_index(sae_acts_king_saved, sae_acts_queen_saved)
    print(cs_list.detach().cpu().numpy())
    
    token_list = model.tokenizer.encode("Queen King Man Woman")
    gamma = getattr(model, model_layer_name).embed_tokens.weight.to(torch.float32)
    print(token_list)
    print(model.tokenizer.decode(token_list))
    gamma_bar = torch.mean(gamma, dim = 0)
    centered_gamma = gamma - gamma_bar
    Cov_gamma = centered_gamma.T @ centered_gamma / gamma.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = gamma @ inv_sqrt_Cov_gamma
    print(f"Embedding shape: {gamma.shape}")
    
    itm1 = gamma[1, :] - gamma[2, :]
    itm2 = gamma[3, :] - gamma[4, :]
    res = torch.matmul(itm1.unsqueeze(0), g.T)
    res = torch.matmul(res, g)
    print(res.shape)
    print(itm2.shape)
    res = torch.nn.functional.cosine_similarity(res.reshape(1, -1), itm2.unsqueeze(0))
    print(res.item())
    exit()
    MW_representation = hidden_states[0][0, -1, :] - hidden_states[0][1, -1, :]
    KQ_representation = hidden_states[0][2, -1, :] - hidden_states[0][3, -1, :]
    print(torch.nn.functional.cosine_similarity(MW_representation.unsqueeze(0), KQ_representation.unsqueeze(0)))
    exit()
    
    for top_k in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]:
        with model.trace("Every kingdom has its ruler, typically a", invoker_args={"truncation": True, "max_length": 1024}):
            hidden_states = getattr(model, model_layer_name).layers[args.model_layer].output
            diff_sae = (sae.W_dec[torch.tensor(list(queen_unique))] - sae.W_dec[torch.tensor(list(king_unique))]).mean(dim=0)
            # diff_sae = (sae.W_dec[torch.tensor(list(queen_unique))]).mean(dim=0)
            gender_feature = torch.nn.functional.normalize(diff_sae, p=2, dim=0)
            getattr(model, model_layer_name).layers[args.model_layer].output[0][:, :, :] = (top_k * gender_feature + hidden_states[0][:, :, :])
            patched_logits = model.lm_head.output.save()
        print(model.tokenizer.decode(patched_logits[0, :, :].argmax(dim=1)))

    heatmap_path = os.path.join(args.log_dir, f"Man_Woman_cosine_similarity_heatmap_layer{args.model_layer}.pdf")
    cs = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            cs[i, j] = torch.nn.functional.cosine_similarity(
                sae.W_dec[torch.tensor(list(queen_unique))[i]].unsqueeze(0), 
                sae.W_dec[torch.tensor(list(king_unique))[j]].unsqueeze(0)
            )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cs.detach().cpu().numpy(), annot=True, fmt='.3f', cmap='RdBu_r', center=0, xticklabels=[f'Woman_{i}' for i in range(3)], yticklabels=[f'Man_{i}' for i in range(3)], cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Cosine Similarity between Man and Woman SAE Features')
    plt.xlabel('Woman Features')
    plt.ylabel('Man Features')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    cs2 = torch.zeros(3, 3)
    for i in range(3):  # diff_sae1 (Man/Woman) combinations
        for j in range(3):  # diff_sae2 (King/Queen) combinations
            # diff_sae1: Woman[i] - Man[i] (using same i for both)
            diff_sae1 = (sae.W_dec[torch.tensor(list(queen_unique))[i]] - sae.W_dec[torch.tensor(list(king_unique))[i]]).unsqueeze(0)
            # diff_sae2: Queen[j] - King[j] (using same j for both)
            diff_sae2 = (sae.W_dec[torch.tensor(list(queen_unique_2))[j]] - sae.W_dec[torch.tensor(list(king_unique_2))[j]]).unsqueeze(0)
            cs2[i, j] = torch.nn.functional.cosine_similarity(diff_sae1, diff_sae2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cs2.detach().cpu().numpy(), 
        annot=True, 
        fmt='.3f', 
        cmap='RdBu_r', 
        center=0, 
        xticklabels=[f'Queen-King_{j}' for j in range(3)], 
        yticklabels=[f'Woman-Man_{i}' for i in range(3)], 
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Cosine Similarity between (Woman-Man) and (Queen-King) Difference Vectors')
    plt.xlabel('Queen-King Difference Vectors')
    plt.ylabel('Woman-Man Difference Vectors')
    plt.savefig(os.path.join(args.log_dir, f"diff_sae1_diff_sae2_cosine_similarity_heatmap_layer{args.model_layer}.pdf"), dpi=300, bbox_inches='tight')
    plt.show()
        
def select_the_different_index(sae_acts_king: torch.Tensor, sae_acts_queen: torch.Tensor, top_k: int = 100) -> tuple[set, set, set, set]:
    """
    Select the different index between two SAE activations.
    
    Args:
        sae_acts_king: SAE activations for "king" input [batch_size, feature_dim]
        sae_acts_queen: SAE activations for "queen" input [batch_size, feature_dim]
    
    Returns:
        tuple containing:
            - king_unique_indices: Indices that are in king's top-k but not in queen's top-k
            - queen_unique_indices: Indices that are in queen's top-k but not in king's top-k
    """
    
    # Get top-k indices for both activations
    king_top_indices = sae_acts_king.topk(top_k).indices.flatten()
    queen_top_indices = sae_acts_queen.topk(top_k).indices.flatten()
    
    min_king_indices = sae_acts_king.topk(top_k, largest=False).indices.flatten()
    min_queen_indices = sae_acts_queen.topk(top_k, largest=False).indices.flatten()
    
    # Convert to sets for easier set operations
    king_set = set(king_top_indices.cpu().numpy())
    queen_set = set(queen_top_indices.cpu().numpy())
    min_king_set = set(min_king_indices.cpu().numpy())
    min_queen_set = set(min_queen_indices.cpu().numpy())
    
    king_unique = king_set - queen_set
    queen_unique = queen_set - king_set
    min_king_unique = min_king_set - min_queen_set
    min_queen_unique = min_queen_set - min_king_set

    return king_unique, queen_unique, min_king_unique, min_queen_unique

if __name__ == "__main__":
    DiffSAE()
