import torch
import transformer_lens
import datasets

class ActivationsStore:
    def __init__(
        self,
        model: transformer_lens.hook_points.HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.dataset = iter(datasets.load_dataset(cfg["dataset"], split="train", streaming=True, trust_remote_code=True))
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "input_ids" in sample:
            return "input_ids"
        elif "output" in sample:
            return "output"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self)->torch.Tensor:
        """Get a batch of tokens from the dataset

        Returns:
            torch.Tensor: Batch of tokens
        """
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text" or self.tokens_column == "output":
                tokens = self.model.to_tokens(batch["text"], truncate=True, move_to_device=True, prepend_bos=True).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor)->torch.Tensor:
        """Get the activations for a batch of tokens

        Args:
            batch_tokens: torch.Tensor: Batch of tokens

        Returns:
            torch.Tensor: Activations for the batch of tokens
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["layer"] +1,
            )
        return cache[self.hook_point]

    def _fill_buffer(self)->torch.Tensor:
        """Fill the activation buffer

        Returns:
            torch.Tensor: Activation buffer
        """
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self)->torch.utils.data.DataLoader:
        """Get a dataloader for the activation buffer

        Returns:
            torch.utils.data.DataLoader: Dataloader for the activation buffer
        """
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self)->torch.Tensor:
        """Get the next batch from the dataloader

        Returns:
            torch.Tensor: Next batch from the dataloader
        """
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]

