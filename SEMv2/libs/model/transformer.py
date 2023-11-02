import math
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    '''
        Self attention layer
    '''
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0

        self.query = nn.Linear(self.hidden_size, self.attention_head_size * self.num_attention_heads)
        self.key = nn.Linear(self.hidden_size, self.attention_head_size * self.num_attention_heads)
        self.value = nn.Linear(self.hidden_size, self.attention_head_size * self.num_attention_heads)

        # self.dropout = nn.Dropout(dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores.float().masked_fill_((1-attention_mask.unsqueeze(1).unsqueeze(1)).to(torch.bool), float(-1e8)) # remove padding token
        
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class FFNIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FFNOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FFNLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super().__init__()
        self.intermediate_layer = FFNIntermediate(hidden_size, intermediate_size)
        self.output_layer = FFNOutput(intermediate_size, hidden_size, dropout_prob)
    
    def forward(self, hidden_states):
        intermediate_output = self.intermediate_layer(hidden_states)
        layer_output = self.output_layer(intermediate_output, hidden_states)
        return layer_output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob):
        super().__init__()
        self.sa_layer = SelfAttentionLayer(hidden_size, num_attention_heads, dropout_prob)
        self.ffn_layer = FFNLayer(hidden_size, intermediate_size, dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.sa_layer(hidden_states, attention_mask)
        hidden_states = self.ffn_layer(hidden_states)
        return hidden_states