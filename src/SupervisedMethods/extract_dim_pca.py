import argparse
import torch
import nnsight
import datasets
import os

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from tqdm import tqdm
from sklearn.decomposition import PCA


def config() -> argparse.Namespace:
    """
    Config for the extract_dim_pca script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        choices=["EleutherAI/pythia-70m", "google/gemma-2-2b", "Qwen/Qwen3-4B-Thinking-2507", "openai-community/gpt2"],
        help="Model to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pyvene/axbench-concept16k_v2",
        choices=["monology/pile-uncopyrighted", "pyvene/axbench-concept16k_v2"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/supervised_methods/",
        help="Directory to save logs",
    )
    parser.add_argument("--model_layer", type=int, default=3, help="Model layer to use")
    parser.add_argument("--debug_mode", type=bool, default=True, help="Debug mode")
    return parser.parse_args()


def load_concept_dataset(dataset_name: str) -> dict:
    """
    Load the concept dataset and return a dictionary of datasets.
    Args:
        dataset_name (str): The name of the dataset to load.
    Returns:
        dataset_dict (dict): A dictionary of datasets.
    """
    dataset_dict = {}
    dataset = datasets.load_dataset(dataset_name, split="train")
    for idx, data in tqdm(enumerate(dataset), desc="Loading dataset"):
        if dataset_dict.get(data["output_concept"]) is None:
            dataset_dict[data["output_concept"]] = []
        elif len(dataset_dict[data["output_concept"]]) > 100:
            continue
        else:
            dataset_dict[data["output_concept"]].append(data["output"])
        if idx > 1100:
            break
    return dataset_dict


def dim_pca_vector(
    hidden_states: torch.Tensor, neg_diff_in_means: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the difference-in-means vector and the PCA of the hidden states.
    Args:
        hidden_states (torch.Tensor[batch_size, seq_len, d_model]): The hidden states of the model.
    Returns:
        diff_in_means (torch.Tensor): The difference-in-means vector.
        pca (torch.Tensor): The PCA of the hidden states.
    """
    # difference-in-means vector
    if neg_diff_in_means is None:
        diff_in_means = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            .mean(dim=0)
            .to(hidden_states.device)
            .float()
        )
        return diff_in_means, None
    else:
        diff_in_means = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            .mean(dim=0)
            .to(hidden_states.device)
            .float()
            - neg_diff_in_means
        )
        # PCA
        pca = PCA(n_components=5).fit(
            hidden_states.reshape(-1, hidden_states.shape[-1])
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        pca_first_5_components = (
            torch.tensor(pca.components_[:5]).to(hidden_states.device).float()
        )
        logger.info(f"PCA first 5 components shape: {pca_first_5_components.shape}")
        logger.info(f"diff_in_means shape: {diff_in_means.shape}")
        return diff_in_means, pca_first_5_components


def extract_dim_pca() -> None:
    """
    Extract the dimension of the dataset using PCA.
    """
    args = config()
    os.makedirs(args.log_dir, exist_ok=True)
    logger.add(
        f"{args.log_dir}/extract_dim_pca_{args.dataset_name.split('/')[-1]}_{args.model_name.split('/')[-1]}_{args.model_layer}.log",
        rotation="100 MB",
        retention="10 days",
    )
    if args.model_name == "EleutherAI/pythia-70m":
        assert args.model_layer < 6
        model_layer_name = "gpt_neox"
    elif args.model_name == "google/gemma-2-2b":
        assert args.model_layer < 12
        model_layer_name = "model"
    elif args.model_name == "Qwen/Qwen3-4B-Thinking-2507":
        assert args.model_layer < 24
        model_layer_name = "model"
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    logger.info(args)
    model = nnsight.LanguageModel(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    dataset_dict = load_concept_dataset(args.dataset_name)
    keys_len = len(dataset_dict.keys()) - 1
    # fig 1 for cos similarity between diff_in_means and pca_first_5_components
    fig, ax = plt.subplots(keys_len, 1, figsize=(10, max(6, 2 * keys_len)))
    if args.model_name == "EleutherAI/pythia-70m":
        neg_diff_in_means = torch.empty(
            getattr(model, model_layer_name).embed_in.weight.shape[1]
        )
    elif args.model_name == "google/gemma-2-2b":
        neg_diff_in_means = torch.empty(
            getattr(model, model_layer_name).embed_tokens.weight.shape[1]
        )
    elif args.model_name == "Qwen/Qwen3-4B-Thinking-2507":
        neg_diff_in_means = torch.empty(
            getattr(model, model_layer_name).embed_tokens.weight.shape[1]
        )
    else: 
        raise ValueError(f"Model {args.model_name} not supported")
    # fig 2 for heatmap of pca_first_5_components
    heatmap_fig, heatmap_ax = plt.subplots(keys_len, 1, figsize=(10, keys_len * 10))
    for idx, (key, value) in enumerate(dataset_dict.items()):
        with model.trace(value):
            # hidden states: tuple of [batch_size, seq_len, d_model]
            hidden_states = (
                getattr(model, model_layer_name).layers[args.model_layer].output.save()
            )
        if args.debug_mode:
            logger.info(f"Hidden states shape: {len(hidden_states)}")
            logger.info(f"Hidden states shape: {hidden_states[0].shape}")

        if idx == 0:
            diff_in_means, pca_first_5_components = dim_pca_vector(
                hidden_states=hidden_states[0], neg_diff_in_means=None
            )
            neg_diff_in_means = diff_in_means
        else:
            diff_in_means, pca_first_5_components = dim_pca_vector(
                hidden_states=hidden_states[0], neg_diff_in_means=neg_diff_in_means
            )

            # plot the cos similarity between diff_in_means and pca_first_5_components
            cos_similarity = torch.nn.functional.cosine_similarity(
                diff_in_means, pca_first_5_components, dim=0
            )
            ax[idx - 1].plot(cos_similarity.detach().cpu().numpy())
            ax[idx - 1].set_title(f"{key}", pad=8, fontsize=10)
            ax[idx - 1].set_ylim(0, 1)
            ax[idx - 1].set_xlim(0, 5)
            for lbl in ax[idx - 1].get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")

            # compute the correlation between pca_first_5_components and diff_in_means
            cos_sim_pca_first_5_components = torch.empty(5, 5)
            for i in range(5):
                for j in range(5):
                    cos_sim_pca_first_5_components[i, j] = (
                        torch.nn.functional.cosine_similarity(
                            pca_first_5_components[i], pca_first_5_components[j], dim=0
                        )
                    )
            if args.debug_mode:
                logger.info(
                    f"Cos similarity between pca_first_5_components and diff_in_means: {cos_sim_pca_first_5_components.shape}"
                )

            # plot the heatmap of pca_first_5_components
            sns.heatmap(
                cos_sim_pca_first_5_components.detach().cpu().numpy(),
                cmap="viridis",
                xticklabels=False,
                yticklabels=[
                    f"PC{i+1}" for i in range(cos_sim_pca_first_5_components.shape[0])
                ],
                cbar_kws={"label": "Value"},
                ax=heatmap_ax[idx - 1],
            )
            heatmap_ax[idx - 1].set_title(f"{key}", pad=8, fontsize=10)
            for lbl in heatmap_ax[idx - 1].get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")
                
    # pca strictly orthogonal
    heatmap_fig.savefig(
        f"{args.log_dir}/heatmap_pca_first_5_components_{args.dataset_name.split('/')[-1]}_{args.model_name.split('/')[-1]}_{args.model_layer}.pdf"
    )
    plt.close(heatmap_fig)

    # save the figure as pdf
    fig.suptitle(
        f"Cos similarity between diff_in_means and pca_first_5_components for {args.dataset_name.split('/')[-1]}",
        y=0.96,
        fontsize=12,
    )
    fig.savefig(
        f"{args.log_dir}/dim_pca_vector_{args.dataset_name.split('/')[-1]}_{args.model_name.split('/')[-1]}_{args.model_layer}.pdf"
    )
    plt.close(fig)


if __name__ == "__main__":
    extract_dim_pca()
