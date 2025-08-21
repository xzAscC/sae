import datasets
import nnsight
import torch
import gc
from dataclasses import dataclass

tracer_kwargs = {"scan": True, "validate": True}


@dataclass
class ActivationBuffer:
    """
    Activation buffer for the residual stream of the model
    """

    data: datasets.Dataset
    model: nnsight.LanguageModel
    model_layer_name: str
    model_layer: int
    d_submodule: int
    n_ctxs: int
    ctx_len: int
    device: str
    batch_size: int
    remove_bos: bool = False
    add_special_token: bool = True

    def __post_init__(self):
        # initialize the activation buffer
        self.read = torch.zeros(0).bool()
        self.activation_buffer_size = int(self.n_ctxs * self.ctx_len)
        self.activations = torch.empty(
            0, self.d_submodule, device=self.device, dtype=self.model.dtype
        )
        self.remove_bos = self.remove_bos and (
            self.model.tokenizer.bos_token_id is not None
        )

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with torch.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[
                torch.randperm(len(unreads), device=unreads.device)[: self.batch_size]
            ]
            self.read[idxs] = True
            return self.activations[idxs]

    def text_batch(self, batch_size=None):
        # TODO: length enough?
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.batch_size
        try:
            texts = []

            while len(texts) < batch_size:
                try:
                    texts.append(next(self.data))
                except StopIteration:
                    if hasattr(self.data, "__iter__") and not hasattr(
                        self.data, "__next__"
                    ):
                        self.data = iter(self.data)
                    else:
                        try:
                            if hasattr(self.data, "_dataset"):
                                self.data = iter(self.data._dataset)
                            else:
                                break
                        except:
                            break

            if not texts:
                raise StopIteration("End of data stream reached")

            return texts
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            add_special_tokens=self.add_special_token,
        )

    def refresh(self):
        """
        Refresh the activation buffer
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = torch.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        while current_idx < self.activation_buffer_size:
            with torch.no_grad():
                tokens = self.tokenized_batch()
                with self.model.trace(
                    tokens,
                    **tracer_kwargs,
                    invoker_args={"truncation": True, "max_length": self.ctx_len},
                ):
                    if self.model_layer_name == "transformer":
                        hidden_states = getattr(self.model, self.model_layer_name).h[self.model_layer].output.save()
                    else:
                        hidden_states = getattr(self.model, self.model_layer_name).layers[self.model_layer].output.save()
                    input = self.model.inputs.save()

                    if self.model_layer_name == "transformer":
                        getattr(self.model, self.model_layer_name).h[self.model_layer].output.stop()
                    else:
                        getattr(self.model, self.model_layer_name).layers[self.model_layer].output.stop()
            mask = input[1]["attention_mask"] != 0
            
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            if self.remove_bos:
                if self.model.tokenizer.bos_token_id is not None:
                    bos_mask = input[1]["input_ids"] == self.model.tokenizer.bos_token_id
                    mask = mask & ~bos_mask
                else:
                    # some models (like Qwen) don't have a bos token, so we need to remove the first non-pad token
                    assert mask.dim() == 2, "expected shape (batch_size, seq_len)"
                    first_one = (mask.to(torch.int64).cumsum(dim=1) == 1) & mask
                    mask = mask & ~first_one

            hidden_states = hidden_states[mask]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)
            
        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "d_dictionary": self.d_dictionary,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "device": self.device,
            "batch_size": self.batch_size,
        }


def load_dataset(dataset_name: str, split: str = "train", streaming: bool = True) -> tuple[datasets.Dataset, str]:
    """
    Load a dataset from HuggingFace and return a generator that yields text.

    Args:
        dataset_name: Name of the dataset (e.g., "pyvene/axbench-concept16k_v2")
        split: Dataset split to use (default: "train")
        streaming: Whether to use streaming mode (default: True)

    Returns:
        Generator that yields text strings
    """
    if dataset_name == "pyvene/axbench-concept16k_v2":
        dataset = datasets.load_dataset("pyvene/axbench-concept16k_v2", split=split, streaming=streaming)
        data_column_name = "output"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    def generator():
        for item in dataset:
            yield item[data_column_name]
    return generator()