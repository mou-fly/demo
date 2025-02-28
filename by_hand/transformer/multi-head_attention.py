import torch
import torch.nn as nn
import math

n_head = 8
d_model = 512
x = torch.randn(128, 64, d_model)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model):


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_h = self.d_model // self.n_head
        self.w_q = nn.Linear(d_model, self.d_h * n_head)
        self.w_k = nn.Linear(d_model, self.d_h * n_head)
        self.w_v = nn.Linear(d_model, self.d_h * n_head)
        self.softmax = nn.Softmax(dim=-1)
        self.concat = nn.Linear(d_model, self.d_h * n_head)

    def forward(self, q, k, v, mask=None):
        batch, seq_len, _ = q.size()
        q = self.w_q(q).view(batch, seq_len, self.n_head, self.d_h).permute(0, 2, 1, 3)
        k = self.w_k(k).view(batch, seq_len, self.n_head, self.d_h).permute(0, 2, 1, 3)
        v = self.w_v(v).view(batch, seq_len, self.n_head, self.d_h).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(self.d_h)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, d_model)
        output = self.concat(score)
        return output


model = MultiHeadAttention(d_model, n_head)
print(model(x, x, x))
