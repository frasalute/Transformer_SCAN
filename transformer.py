import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.query_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, emb_dim)

    def _split_heads(self, hidden_states):
        batch_size, seq_len, emb_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        return hidden_states.permute(0, 2, 1, 3)

    def _merge_heads(self, hidden_states):
        batch_size, num_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return hidden_states

    def forward(self, query, key, value, mask=None):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attn_score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn = attn_weights @ value

        attn = self._merge_heads(attn)
        attn_output = self.out_proj(attn)

        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        # Self-Attention
        attn_output = self.attn(query, key, value, mask)
        attn_output = self.dropout1(attn_output)
        attn_output = self.norm1(query + attn_output)  # Skip connection

        # Feedforward
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout2(ffn_output)
        output = self.norm2(attn_output + ffn_output)  # Skip connection

        return output


def get_sinusoid_table(max_len, emb_dim):
    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(0, emb_dim, 2):
            angle = pos / (10000 ** ((2 * (i // 2)) / emb_dim))
            sinusoid_table[pos, i] = math.sin(angle)
            if i + 1 < emb_dim:
                sinusoid_table[pos, i + 1] = math.cos(angle)
    return sinusoid_table


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        sinusoid_table = get_sinusoid_table(max_len, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout, forward_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_len = x.size(1)
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.dropout(tok_emb + pos_emb.unsqueeze(0))

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim),
        )
        self.norm3 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        self_attn_output = self.dropout1(self_attn_output)
        self_attn_output = self.norm1(x + self_attn_output)  # Skip connection

        # Cross-Attention
        cross_attn_output = self.cross_attn(self_attn_output, enc_output, enc_output, src_mask)
        cross_attn_output = self.dropout2(cross_attn_output)
        cross_attn_output = self.norm2(self_attn_output + cross_attn_output)  # Skip connection

        # Feedforward
        ffn_output = self.ffn(cross_attn_output)
        ffn_output = self.dropout3(ffn_output)
        output = self.norm3(cross_attn_output + ffn_output)  # Skip connection

        return output


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        sinusoid_table = get_sinusoid_table(max_len, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        seq_len = x.size(1)
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.dropout(tok_emb + pos_emb.unsqueeze(0))

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.fc_out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.1,
        max_len=128,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def create_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        no_peak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_mask & no_peak_mask.unsqueeze(0)

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        return dec_output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        src_vocab_size=200,
        tgt_vocab_size=220,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    # source input: batch size 4, sequence length of 75
    src_in = torch.randint(0, 200, (4, 75)).to(device)

    # target input: batch size 4, sequence length of 80
    tgt_in = torch.randint(0, 220, (4, 80)).to(device)

    # expected output shape of the model
    expected_out_shape = torch.Size([4, 80, 220])

    with torch.no_grad():
        out = model(src_in, tgt_in)

    assert (
        out.shape == expected_out_shape
    ), f"wrong output shape, expected: {expected_out_shape}"

    print("Passed test!")


if __name__ == "__main__":
    main()