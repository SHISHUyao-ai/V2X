"""
Who2com 完整三阶段握手通信机制

论文 Figure 3 完整流程：
  Phase 1 - Request:  退化智能体广播低维消息
  Phase 2 - Match:    正常智能体返回匹配分数
  Phase 3 - Connect:  退化智能体选择最优智能体并获取特征

训练时：softmax 加权所有agent（Eq 5）
推理时：argmax 或动态K 选择最优agent（Eq 6）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generators import MessageGenerator, KeyGenerator
from .attention import GeneralAttention


class Who2comHandshake(nn.Module):
    """
    完整的 Who2com 三阶段握手
    
    用法：
        model = Who2comHandshake(feature_dim=256, message_dim=8, key_dim=1024)
        
        # 训练
        weights, fused = model(degraded_feat, normal_feats, training=True)
        
        # 推理
        weights, fused, selected_idx = model(degraded_feat, normal_feats, training=False)
    """
    
    def __init__(self, feature_dim, message_dim=8, key_dim=1024):
        """
        Args:
            feature_dim: backbone 输出的特征通道数
            message_dim: 请求消息维度（论文推荐8）
            key_dim: key维度（论文推荐1024）
        """
        super().__init__()
        
        # Phase 1: Request - 退化智能体生成低维请求
        self.message_generator = MessageGenerator(feature_dim, message_dim)
        
        # Phase 2: Match - 正常智能体生成key + 匹配
        self.key_generator = KeyGenerator(feature_dim, key_dim)
        self.attention = GeneralAttention(message_dim, key_dim)
        
        # 保存维度信息
        self.feature_dim = feature_dim
        self.message_dim = message_dim
        self.key_dim = key_dim
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """He初始化，确保训练稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, degraded_feat, normal_feats, training=True, top_k=None):
        """
        Args:
            degraded_feat: (B, C, H, W)  退化智能体的特征图
            normal_feats:  List[Tensor(B, C, H, W)]  正常智能体的特征图列表
            training:      训练/推理模式
            top_k:         推理时选几个agent（None=动态K, 1=论文原始做法）
        
        Returns:
            weights:       (B, N)  每个agent的权重
            fused_feat:    (B, C, H, W)  融合后的特征
            selected_idx:  (仅在推理时返回) 选中的agent索引
        """
        N = len(normal_feats)
        
        if N == 0:
            return (
                torch.zeros(degraded_feat.size(0), 0, device=degraded_feat.device),
                degraded_feat,
                None
            )
        
        # =============================================
        # Phase 1: REQUEST
        # 论文 Eq(1): μ_j = G_m(x̃_j; θ_m)
        # =============================================
        message = self.message_generator(degraded_feat)  # (B, message_dim)
        
        # =============================================
        # Phase 2: MATCH
        # 论文 Eq(2): s_ji = Φ(μ_j, κ_i)
        # =============================================
        keys = self.key_generator(normal_feats)  # (B, N, key_dim)
        scores = self.attention(message, keys)    # (B, N)
        
        # =============================================
        # Phase 3: CONNECT
        # 训练: 论文 Eq(5) softmax加权所有agent
        # 推理: 论文 Eq(6) argmax选最高分
        # =============================================
        
        if training:
            # 论文原文: α_j = softmax([s_j1; ...; s_jN])
            weights = F.softmax(scores, dim=-1)  # (B, N)
            
            # f_sum = Σ α_j,i · f_i
            fused_feat = torch.zeros_like(degraded_feat)
            for i, feat in enumerate(normal_feats):
                w = weights[:, i].view(-1, 1, 1, 1)
                fused_feat = fused_feat + w * feat
            
            return weights, fused_feat, None
        
        else:
            # =============================================
            # 推理模式
            # =============================================
            weights = F.softmax(scores, dim=-1)  # (B, N)
            B = degraded_feat.size(0)
            
            if top_k is None:
                # 动态K: 选所有分数 > 阈值的agent
                threshold = 1.0 / N
                selected_mask = weights > threshold
                selected_idx = selected_mask.nonzero(as_tuple=False)
            
            elif top_k == 1:
                # 论文原始做法: argmax → (B,)
                selected_idx = scores.argmax(dim=-1)
                
                # 只融合每个batch选中的那一个agent
                fused_feat = torch.zeros_like(degraded_feat)
                for b in range(B):
                    idx = selected_idx[b].item()
                    fused_feat[b:b+1] = normal_feats[idx][b:b+1]
            
            else:
                # 固定 top-k → (B, K)
                _, selected_idx = scores.topk(top_k, dim=-1)
                
                fused_feat = torch.zeros_like(degraded_feat)
                for b in range(B):
                    for k_idx in selected_idx[b]:
                        idx = k_idx.item()
                        fused_feat[b:b+1] = fused_feat[b:b+1] + normal_feats[idx][b:b+1]
                    fused_feat[b:b+1] = fused_feat[b:b+1] / top_k
            
            # 如果 top_k != 1，上面的逻辑已经在循环里处理了 fused_feat
            # 但 top_k=None 走动态K时 fused_feat 未初始化
            if top_k is None:
                fused_feat = torch.zeros_like(degraded_feat)
                for b in range(B):
                    b_indices = selected_idx[selected_idx[:, 0] == b][:, 1]
                    if len(b_indices) == 0:
                        fused_feat[b:b+1] = degraded_feat[b:b+1]
                    else:
                        for idx in b_indices:
                            fused_feat[b:b+1] = fused_feat[b:b+1] + normal_feats[idx.item()][b:b+1]
                        fused_feat[b:b+1] = fused_feat[b:b+1] / len(b_indices)
            
            return weights, fused_feat, selected_idx
    
    def get_bandwidth_usage(self, N_agents):
        """
        计算通信带宽使用量
        
        论文指标:
        - message:   message_dim × 32bit = 极少（如 8×32=256bit）
        - key:       不需要传输（本机计算）
        - feature:   1个agent的特征图大小
        
        对比:
        - CatAll:    N × feature_size → 线性增长
        - Ours:      1 × feature_size → 恒定
        """
        message_bits = self.message_dim * 32
        feature_bits = self.feature_dim * 16 * 16 * 32  # 假设16×16特征图
        
        # 我们的带宽 = message广播 + 1个agent的特征
        our_bw = (message_bits * N_agents + feature_bits) / 8 / 1024  # KB
        
        # CatAll 基线 = N个agent的特征
        catall_bw = (feature_bits * N_agents) / 8 / 1024
        
        return {
            'ours_kb': our_bw,
            'catall_kb': catall_bw,
            'ratio': our_bw / catall_bw if catall_bw > 0 else 0
        }