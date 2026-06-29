"""
Who2com 论文中的 Message Generator 和 Key Generator
论文 Eq(1): μ_j = G_m(x̃_j; θ_m) → 低维请求向量
论文 Eq(2): κ_i  = G_k(x_i; θ_k)  → 高维key向量
"""

import torch
import torch.nn as nn


class MessageGenerator(nn.Module):
    """
    论文 Eq(1): 退化智能体把它的观察压缩成低维请求消息
    
    设计原则（来自论文 Figure 5a）：
    - message 可以非常小（8维就够用）
    - 这是发出去的数据，越小越省带宽
    """
    
    def __init__(self, feature_dim, message_dim=8):
        """
        Args:
            feature_dim: 输入特征维度（如 backbone 输出的通道数）
            message_dim: 输出的消息维度（论文推荐 8，实验范围 1~64）
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 空间池化：把 (C, H, W) 压成 (C,)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),  # (B, C)
            
            # 降维到 message_dim
            nn.Linear(feature_dim, message_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(message_dim * 2, message_dim),
            nn.ReLU(inplace=True),
        )
        
        self.message_dim = message_dim
    
    def forward(self, degraded_obs):
        """
        Args:
            degraded_obs: (B, C, H, W)  退化智能体的特征图
        
        Returns:
            message: (B, message_dim)  压缩后的请求消息
        """
        return self.encoder(degraded_obs)


class KeyGenerator(nn.Module):
    """
    论文 Eq(2): 每个正常智能体生成自己的 key
    
    设计原则（来自论文 Figure 5b）：
    - key 可以很大（1024维），因为只在本机算，不需要传输
    - 大 key 能保留更多信息，匹配更准
    """
    
    def __init__(self, feature_dim, key_dim=1024):
        """
        Args:
            feature_dim: 输入特征维度
            key_dim: 输出的 key 维度（论文推荐 1024）
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),  # (B, C)
            
            nn.Linear(feature_dim, key_dim),
            nn.ReLU(inplace=True),
            nn.Linear(key_dim, key_dim),
            nn.ReLU(inplace=True),
        )
        
        self.key_dim = key_dim
    
    def forward(self, normal_obs):
        """
        Args:
            normal_obs: (B, N, C, H, W) 或 List of (B, C, H, W)
        
        Returns:
            keys: (B, N, key_dim)
        """
        if isinstance(normal_obs, list):
            keys = torch.stack([self.encoder(obs) for obs in normal_obs], dim=1)
        elif normal_obs.dim() == 5:
            # (B, N, C, H, W)
            B, N, C, H, W = normal_obs.shape
            normal_obs = normal_obs.view(B * N, C, H, W)
            keys = self.encoder(normal_obs)  # (B*N, key_dim)
            keys = keys.view(B, N, -1)       # (B, N, key_dim)
        else:
            # (B, C, H, W) - 单个agent
            keys = self.encoder(normal_obs).unsqueeze(1)
        
        return keys