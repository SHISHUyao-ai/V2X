"""
Who2com 论文中的匹配函数实现
论文 Eq(3): General Attention  Φ = μ^T · W_a · κ
还实现了 Scaled Dot-Product 和 Additive 作为对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralAttention(nn.Module):
    """
    论文 Eq(3): Φ(μ, κ) = μ^T · W_a · κ
    
    核心优势：message 和 key 的维度可以不同（非对称压缩）
    message 可以极小（8维），key 可以很大（1024维）
    """
    
    def __init__(self, message_dim, key_dim):
        super().__init__()
        # W_a: (message_dim, key_dim)
        self.W_a = nn.Parameter(torch.randn(message_dim, key_dim) * 0.01)
    
    def forward(self, message, keys):
        """
        Args:
            message: (batch, message_dim)  请求向量
            keys:    (batch, num_agents, key_dim)  各agent的key
        
        Returns:
            scores:  (batch, num_agents)  匹配分数（未归一化）
        """
        # message: (B, m) → (B, 1, m)
        # W_a:     (m, k)
        # message @ W_a: (B, 1, k)
        transformed = torch.matmul(message.unsqueeze(1), self.W_a)  # (B, 1, k)
        
        # transformed @ keys^T: (B, 1, k) @ (B, k, N) → (B, 1, N) → (B, N)
        scores = torch.bmm(transformed, keys.transpose(1, 2)).squeeze(1)
        
        return scores


class ScaledDotProductAttention(nn.Module):
    """
    Φ = μ^T · κ / √d_n
    要求 message 和 key 维度相同
    """
    
    def forward(self, message, keys):
        d_n = message.size(-1)
        scores = torch.bmm(message.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        return scores / (d_n ** 0.5)


class AdditiveAttention(nn.Module):
    """
    Φ = W_a^T · tanh(W_k · κ + W_m · μ)
    允许不同维度
    """
    
    def __init__(self, message_dim, key_dim, hidden_dim=128):
        super().__init__()
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(message_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, message, keys):
        """
        message: (B, m)
        keys:    (B, N, k)
        """
        B, N, k = keys.shape
        
        # keys → (B, N, hidden)
        k_transformed = self.W_k(keys)
        # message → (B, 1, hidden) → (B, N, hidden)
        m_transformed = self.W_m(message).unsqueeze(1).expand(-1, N, -1)
        
        # tanh(W_k κ + W_m μ)
        combined = torch.tanh(k_transformed + m_transformed)
        
        # W_a^T · tanh(...) → (B, N, 1) → (B, N)
        scores = self.W_a(combined).squeeze(-1)
        
        return scores