import torch
import torch.nn as nn

class PMCTNetwork_attention(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=1024, num_blocks: int = 4, token_size: int = 8, hidden_fc: int = None):
        super(PMCTNetwork_attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.token_size = token_size

        if hidden_fc is None:
            hidden_fc = hidden_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        assert hidden_dim % token_size == 0, "hidden_dim must be divisible by token_size"
        self.token_dim = hidden_dim // token_size
        self.tokenizer = nn.Linear(hidden_dim, token_size * self.token_dim)


        self.pos_emb = nn.Parameter(torch.randn(1, token_size, self.token_dim) * 0.02)


        default_heads = 8
        if self.token_dim % default_heads != 0:
            for h in range(default_heads, 0, -1):
                if self.token_dim % h == 0:
                    default_heads = h
                    break

        self.attention = nn.MultiheadAttention(embed_dim=self.token_dim, num_heads=default_heads, dropout=0.15, batch_first=True)
        self.attn_ln = nn.LayerNorm(self.token_dim)

        class ResidualMLPToken(nn.Module):
            def __init__(self, dim, hidden_inner=None):
                super().__init__()
                if hidden_inner is None:
                    hidden_inner = max(dim // 2, 64)
                self.fc1 = nn.Linear(dim, hidden_inner)
                self.ln1 = nn.LayerNorm(hidden_inner)
                self.act = nn.LeakyReLU()
                self.fc2 = nn.Linear(hidden_inner, dim)
                self.ln2 = nn.LayerNorm(dim)

            def forward(self, x):
                identity = x
                out = self.fc1(x)
                out = self.ln1(out)
                out = self.act(out)
                out = self.fc2(out)
                out = self.ln2(out)
                out = out + identity
                out = self.act(out)
                return out

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualMLPToken(self.token_dim))
        self.token_blocks = nn.Sequential(*blocks)

        aggregated_dim = token_size * self.token_dim
        self.mlp = nn.Sequential(
            nn.Linear(aggregated_dim, hidden_fc // 2),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_fc // 2),
            nn.Dropout(0.25),
            nn.Linear(hidden_fc // 2, max(32, hidden_fc // 4)),
            nn.LeakyReLU(),
            nn.LayerNorm(max(32, hidden_fc // 4)),
            nn.Dropout(0.25),
            nn.Linear(max(32, hidden_fc // 4), 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.06)
            elif 'bias' in name:
                nn.init.zeros_(param)

        with torch.no_grad():

            last_linear = None
            for m in reversed(list(self.mlp)):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
            if last_linear is not None:
                last_linear.bias.fill_(11)
                if last_linear.weight is not None:
                    last_linear.weight.data.mul_(0.04)

    def forward(self, x: torch.Tensor) :

        out = self.input_proj(x)

        tokens = self.tokenizer(out)

        tokens = tokens.view(tokens.size(0), self.token_size, self.token_dim)

        tokens = tokens + self.pos_emb

        attn_out, _ = self.attention(tokens, tokens, tokens)
        attn_out = self.attn_ln(attn_out + tokens)

        attn_out = self.token_blocks(attn_out)

        agg = attn_out.view(attn_out.size(0), -1)
        q_value = self.mlp(agg).squeeze(-1)

        msc_pred = self._get_msc_pred(q_value)
        return q_value, msc_pred

    def _get_msc_pred(self, q_values: torch.Tensor) -> torch.Tensor:
        msc_float = q_values
        mask = msc_float >= 10.4
        result = torch.where(mask, torch.tensor(11.0).to(q_values.device), msc_float)
        result = torch.where(~mask, torch.floor(result), result)
        msc_pred = torch.clamp(result, min=1, max=11).long()
        return msc_pred