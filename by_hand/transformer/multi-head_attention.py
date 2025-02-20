import torch
import torch.functional as F
from torch import nn
import math

test = torch.randn(128, 64, 512)

d_model = 512
n_head = 8


class multi_haed_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_haed_attention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, time, dimension = q.shape
        n_d = self.d_model / self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch_size, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch_size, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch_size, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        mask = torch.tril(torch.ones_like(time, time, dtype=bool), diagonal=0)
        score = score.masked_fill(mask, -math.inf)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, time, dimension)

        output = self.w_combine(score)
        return output


attention = multi_haed_attention(d_model, n_head)
output = attention(test, test, test)
print(output)
