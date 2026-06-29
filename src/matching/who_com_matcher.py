"""
简化接口：封装 Who2comHandshake，提供与 OpenCOOD 兼容的接口
"""

import torch
import torch.nn as nn
from .handshake import Who2comHandshake


class WhoComMatcher(nn.Module):
    """简化版：与 OpenCOOD 接口兼容"""
    
    def __init__(self, feature_dim, message_dim=8, key_dim=1024):
        super().__init__()
        self.handshake = Who2comHandshake(feature_dim, message_dim, key_dim)
    
    def forward(self, ego_feat, cand_feats, training=True):
        weights, fused, selected = self.handshake(ego_feat, cand_feats, training)
        return weights, fused


class WhoComMatcherDynamicK(nn.Module):
    """带动态K的版本"""
    
    def __init__(self, feature_dim, message_dim=8, key_dim=1024):
        super().__init__()
        self.handshake = Who2comHandshake(feature_dim, message_dim, key_dim)
    
    def forward(self, ego_feat, cand_feats, training=True):
        return self.handshake(ego_feat, cand_feats, training, top_k=None)