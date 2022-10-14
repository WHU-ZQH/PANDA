import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)
            # state_dict = torch.load("checkpoints/rte-bert-prefix/prefix_encoder.bin")
            # self.embedding = torch.nn.Embedding.from_pretrained(state_dict.get("embedding.weight"), freeze=False)
        self._reset_params(config)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

    def _reset_params(self, config):
        if config.init_type == "normal":
            torch.nn.init.normal_(self.embedding.weight)
        elif config.init_type == "uniform":
            torch.nn.init.uniform_(self.embedding.weight)
        elif config.init_type == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif config.init_type == "xavier_normal":
            torch.nn.init.xavier_normal_(self.embedding.weight)
        elif config.init_type == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight)
        elif config.init_type == "kaiming_normal":
            torch.nn.init.kaiming_normal_(self.embedding.weight)
        elif config.init_type == "ones":
            torch.nn.init.ones_(self.embedding.weight)
        elif config.init_type == "zeros":
            torch.nn.init.zeros_(self.embedding.weight)
        elif config.init_type == "sparse":
            torch.nn.init.sparse_(self.embedding.weight, 0.5)
        else:
            pass