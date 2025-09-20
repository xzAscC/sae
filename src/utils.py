import torch
import numpy as np
import random
import sae_lens
import transformer_lens

def seed_setup(seed: int) -> None:
    """
    Seed setup

    Args:
        seed (int): Seed for random number generator
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return None


def extract_top_activations_and_cosine_similarity(sae: sae_lens.SAE, model: transformer_lens.HookedTransformer, model_layer_name: str, model_layer, data: list[str], targer_index: int, topk: int) -> torch.Tensor:
    """
    Extract the top activations and cosine similarity between the top activations and the original activations.
    """
    cache = model.run_with_cache(data, names_filter=[model_layer_name], stop_at_layer=model_layer + 1)
    topk_activations = cache[model_layer_name][0][:, targer_index, :].topk(topk).indices
    # sae_weights = sae.W_dec[targer_index]
    # cosine_similarity = torch.nn.functional.cosine_similarity(sae_weights, topk_activations)
    return topk_activations