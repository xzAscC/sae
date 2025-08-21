import torch
import einops
import datasets
import os
import sae_lens
import mlflow
import nnsight
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, overload
from contextlib import nullcontext
from loguru import logger
from typing import Callable
from tqdm import tqdm

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


def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None,
) -> Callable[[int], float]:
    # TODO
    """
    Creates a learning rate schedule function with linear warmup followed by an optional decay phase.

    Note: resample_steps creates a repeating warmup pattern instead of the standard phases, but
    is rarely used in practice.

    Args:
        total_steps: Total number of training steps
        warmup_steps: Steps for linear warmup from 0 to 1
        decay_start: Optional step to begin linear decay to 0
        resample_steps: Optional period for repeating warmup pattern
        sparsity_warmup_steps: Used for validation with decay_start

    Returns:
        Function that computes LR scale factor for a given step
    """
    if decay_start is not None:
        assert (
            resample_steps is None
        ), "decay_start and resample_steps are currently mutually exclusive."
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
        if sparsity_warmup_steps is not None:
            assert (
                decay_start > sparsity_warmup_steps
            ), "decay_start must be > sparsity_warmup_steps."

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    if resample_steps is None:

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                # Warm-up phase
                return step / warmup_steps

            if decay_start is not None and step >= decay_start:
                # Decay phase
                return (total_steps - step) / (total_steps - decay_start)

            # Constant phase
            return 1.0

    else:
        assert (
            0 < resample_steps < total_steps
        ), "resample_steps must be > 0 and < steps."

        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)

    return lr_schedule


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


def get_norm_factor(data, steps: int) -> float:
    # TODO
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147

    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(
        tqdm(data, total=steps, desc="Calculating norm factor")
    ):
        if step > steps:
            break

        count += 1
        mean_squared_norm = torch.mean(torch.sum(act_BD**2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = torch.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")

    return norm_factor


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
        self.register_buffer(
            "threshold_buffer", torch.tensor(-1.0, dtype=torch.float32)
        )

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

    @property
    def config(self):
        return {
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "k": self.k,
            "threshold": self.threshold,
        }

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
            encoded_acts_BF = pre_activation_BF * (pre_activation_BF > self.threshold)

        post_topk = pre_activation_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(pre_activation_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return (
                encoded_acts_BF,
                tops_acts_BK,
                top_indices_BK,
                pre_activation_BF,
            )  # Return pre-ReLU
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    @overload
    def from_pretrained(self, *args, **kwargs):
        pass

    @overload
    def from_pretrained(
        self,
        use_saelens: bool,
        model_name: str,
        model_layer: int,
        device: Optional[str] = None,
    ):
        pass

    def from_pretrained(
        self,
        use_saelens: bool,
        model_name: str,
        model_layer: int,
        trainer: int = 5,
        device: Optional[str] = None,
    ):
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

    def train(
        self,
        data: torch.Tensor,
        warmup_steps: int,
        sparsity_warmup_steps: int,
        save_dir: str,
        save_steps: list[int],
        steps: int,
        device: str,
        lr: Optional[float] = 1e-4,
        auxk_alpha: float = 1 / 32,  # see Appendix A.2
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        normalize_activations: bool = True,
        k_anneal_steps: Optional[int] = None,
        name: str = "SAE",
    ):

        # setup
        autocast_context = (
            nullcontext()
            if device == "cpu"
            else torch.autocast(device_type=device, dtype=torch.bfloat16)
        )
        self.top_k_aux = self.activation_dim // 2  # Heuristic from of topk paper
        self.num_tokens_since_fired = torch.zeros(
            self.dict_size, dtype=torch.long, device=device
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
        self.dead_feature_threshold = 10_000_000
        self.effective_l0 = -1
        self.dead_features = -1
        self.k_anneal_steps = k_anneal_steps
        self.pre_norm_auxk_loss = -1
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta
        self.save_dir = save_dir
        self.auxk_alpha = auxk_alpha
        if normalize_activations:
            norm_factor = get_norm_factor(data, steps=100)
            self.scale_biases(1.0)

        with mlflow.start_run(run_name=f"SAE_{name}"):
            mlflow.log_params(self.config)

            with autocast_context:
                for step, act in enumerate(tqdm(data, total=steps)):
                    act = act.to(dtype=torch.bfloat16)
                    if normalize_activations:
                        act = act / norm_factor

                    if step > steps:
                        break
                    # TODO: for epoch?

                    # training
                    step_losses = self.update_step(step, act, device, optimizer, scheduler)

                    if save_steps and step in save_steps:
                        self.save_checkpoint(step)

                    scheduler.step()

                    mlflow.log_metric("loss", step_losses, step=step)
                    mlflow.log_metric("effective_l0", self.effective_l0, step=step)
                    mlflow.log_metric("dead_features", self.dead_features, step=step)
                    mlflow.log_metric(
                        "pre_norm_auxk_loss", self.pre_norm_auxk_loss, step=step
                    )

                mlflow.end_run()
                logger.info(f"Training complete. Final loss: {step_losses}")

                # Save final model as safetensors
                final_path = os.path.join(self.save_dir, "SAE_final.safetensors")
                try:
                    from safetensors.torch import save_file

                    save_file(self.state_dict(), final_path)
                    logger.info(f"Saved final SAE as safetensors to {final_path}")
                except ImportError:
                    # Fallback to regular torch.save if safetensors not available
                    torch.save(
                        self.state_dict(), os.path.join(self.save_dir, "SAE_final.pt")
                    )
                    logger.warning(
                        "safetensors not available, falling back to torch.save"
                    )
                    logger.info(f"Saved final SAE to {save_dir}")

    def update_step(self, step: int, act: torch.Tensor, device: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> float:
        """
        Update the step and return the losses.
        """
        if step == 0:
            median = self.geometric_median(act)
            median = median.to(self.b_dec.dtype)
            self.b_dec.data = median

        act = act.to(device)
        loss = self.loss(act, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.decoder.weight,
            self.decoder.weight.grad,
            self.activation_dim,
            self.dict_size,
        )
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # do a training step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        self.update_annealed_k(step, self.activation_dim, self.k_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, self.activation_dim, self.dict_size
        )

        return loss.item()

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert (
            0 <= k_anneal_steps < steps
        ), "k_anneal_steps must be >= 0 and < steps."
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.k.fill_(int(annealed_value))

    def loss(self, x: torch.Tensor, step: int):
        # Run the SAE
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.encode(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.decode(f)

        # Measure goodness of reconstruction
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
            if self.auxk_alpha > 0
            else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss

        return loss

    def update_threshold(self, top_acts_BK: torch.Tensor):
        """update the threshold for the SAE

        Args:
            top_acts_BK (t.Tensor): top activations
        """
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=torch.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.threshold < 0:
                self.threshold = min_activation
            else:
                self.threshold = (self.threshold_beta * self.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    @torch.no_grad()
    def geometric_median(self, points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
        """Compute the geometric median `points`. Used for initializing decoder bias."""
        # Initialize our guess as the mean of the points
        guess = points.mean(dim=0)
        prev = torch.zeros_like(guess)

        # Weights for iteratively reweighted least squares
        weights = torch.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess

            # Compute the weights
            weights = 1 / torch.norm(points - guess, dim=1)

            # Normalize the weights
            weights /= weights.sum()

            # Compute the new geometric median
            guess = (weights.unsqueeze(1) * points).sum(dim=0)

            # Early stopping condition
            if torch.norm(guess - prev) < tol:
                break

        return guess

    def save_checkpoint(self, step: int):
        """
        Save the current state of the SAE to a file.
        """
        if not os.path.exists(os.path.join(self.save_dir, "checkpoints")):
            os.makedirs(os.path.join(self.save_dir, "checkpoints"))
        # Save as safetensors format
        checkpoint_path = os.path.join(
            self.save_dir, "checkpoints", f"SAE_checkpoint_step_{step}.safetensors"
        )
        try:
            from safetensors.torch import save_file

            save_file(self.state_dict(), checkpoint_path)
            logger.info(
                f"Saved checkpoint as safetensors at step {step}: {checkpoint_path}"
            )
        except ImportError:
            # Fallback to regular torch.save if safetensors not available
            torch.save(
                self.state_dict(),
                os.path.join(self.save_dir, "checkpoints", f"SAE_checkpoint_step_{step}.pt"),
            )
            logger.warning("safetensors not available, falling back to torch.save")
            logger.info(f"Saved checkpoint at step {step}")

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale
