from matplotlib import use
import torch
import einops
import datasets
import sae_lens
from abc import ABC, abstractmethod
from typing import Optional, overload

class Dictionary(ABC, torch.nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass
    
# The next two functions could be replaced with the ConstrainedAdam Optimizer
@torch.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: torch.nn.Parameter, activation_dim: int, d_sae: int
) -> torch.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: torch.Tensor,
    W_dec_DF_grad: torch.Tensor,
    activation_dim: int,
    d_sae: int,
) -> torch.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape
    assert D == activation_dim
    assert F == d_sae

    normed_W_dec_DF = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)

    parallel_component = einops.einsum(
        W_dec_DF_grad,
        normed_W_dec_DF,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    W_dec_DF_grad -= einops.einsum(
        parallel_component,
        normed_W_dec_DF,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return W_dec_DF_grad

class TopKSAE(Dictionary, torch.nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    """
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        
        # TODO: register buffer
        self.register_buffer("k_buffer", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold_buffer", torch.tensor(-1.0, dtype=torch.float32))

        # decoder shape: (activation_dim, dict_size)
        self.decoder = torch.nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        # encoder shape: (dict_size, activation_dim)
        self.encoder = torch.nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()

        self.b_dec = torch.nn.Parameter(torch.zeros(activation_dim))
    
    @property
    def k(self):
        return self.k_buffer.item()
    
    @property
    def threshold(self):
        return self.threshold_buffer.item()
    @threshold.setter
    def threshold(self, threshold):
        self.threshold_buffer.data = torch.tensor(threshold, dtype=torch.float32)
    
    @k.setter
    def k(self, k):
        self.k_buffer.data = torch.tensor(k, dtype=torch.int)
        
        
    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF
    
    def encode(
        self, x: torch.Tensor, return_topk: bool = False, use_threshold: bool = False
    ):
        pre_activation_BF = self.encoder(x - self.b_dec)

        if use_threshold:
            encoded_acts_BF = pre_activation_BF * (
                pre_activation_BF > self.threshold
            )

        post_topk = pre_activation_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(pre_activation_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, pre_activation_BF  # Return pre-ReLU
        else:
            return encoded_acts_BF
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec
    
    @overload
    def from_pretrained(self, *args, **kwargs):
        pass

    @overload
    def from_pretrained(self, use_saelens: bool, model_name: str, model_layer: int, device: Optional[str] = None):
        pass
    
    def from_pretrained(self, use_saelens: bool, model_name: str, model_layer: int, trainer: int=5, device: Optional[str] = None):
        """
        Load a pretrained autoencoder from a file.
        """
        if model_name == "sae_bench_gemma-2-2b_topk_width-2pow12_date-1109":
            sae = sae_lens.SAE.from_pretrained(
                model_name, f"blocks.{model_layer}.hook_resid_post__trainer_{trainer}"
            )
            self.decoder.weight.data = sae.W_dec.data.T
            self.encoder.weight.data = sae.W_enc.data.T
            self.b_dec.data = sae.b_dec.data
            self.encoder.bias.data = sae.b_enc.data
            del sae
        elif model_name == "sae_bench_pythia70m_sweep_gated_ctx128_0730":
            sae = sae_lens.SAE.from_pretrained(
                model_name, f"blocks.{model_layer}.hook_resid_post__trainer_0"
            )
            self.decoder.weight.data = sae.W_dec.data.T
            self.encoder.weight.data = sae.W_enc.data.T
            self.b_dec.data = sae.b_dec.data
            self.encoder.bias.data = sae.b_enc.data
            del sae
        return self
    