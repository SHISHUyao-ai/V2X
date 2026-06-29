"""测试完整的 Who2com 三阶段握手流程"""
import torch
import sys
sys.path.insert(0, '.')
from src.matching.handshake import Who2comHandshake


def test_full_handshake_training():
    """测试训练模式：softmax加权所有agent"""
    print("=" * 50)
    print("测试 1: 完整三阶段握手（训练模式）")
    
    model = Who2comHandshake(feature_dim=256, message_dim=8, key_dim=1024)
    model.train()
    
    degraded = torch.randn(4, 256, 16, 16)
    normals = [torch.randn(4, 256, 16, 16) for _ in range(5)]
    
    weights, fused, selected = model(degraded, normals, training=True)
    
    assert weights.shape == (4, 5), f"weights: {weights.shape}"
    assert fused.shape == (4, 256, 16, 16), f"fused: {fused.shape}"
    assert selected is None, "训练模式不应返回selected"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4)), "weights应归一化"
    
    print(f"  weights shape: {weights.shape}")
    print(f"  weights[0]: {weights[0].detach().round(decimals=3)}")
    print(f"  weights sum: {weights[0].sum():.3f}")
    print(f"  fused shape: {fused.shape}")
    print("  ✅ 通过")


def test_full_handshake_inference():
    """测试推理模式：argmax选最优"""
    print("\n测试 2: 推理模式（argmax）")
    
    model = Who2comHandshake(feature_dim=256, message_dim=8, key_dim=1024)
    model.eval()
    
    degraded = torch.randn(4, 256, 16, 16)
    normals = [torch.randn(4, 256, 16, 16) for _ in range(5)]
    
    weights, fused, selected = model(degraded, normals, training=False, top_k=1)
    
    assert selected is not None, "推理模式应返回selected"
    print(f"  选中 agent: {selected.tolist()}")
    print(f"  weights: {weights[0].detach().round(decimals=3)}")
    print(f"  fused shape: {fused.shape}")
    print("  ✅ 通过")


def test_gradient_through_handshake():
    """论文关键：梯度能否通过匹配分数回传"""
    print("\n测试 3: 端到端梯度回传")
    
    model = Who2comHandshake(feature_dim=128, message_dim=8, key_dim=256)
    model.train()
    
    degraded = torch.randn(2, 128, 8, 8, requires_grad=True)
    normals = [torch.randn(2, 128, 8, 8, requires_grad=True) for _ in range(3)]
    
    weights, fused, _ = model(degraded, normals, training=True)
    
    # 模拟下游任务loss
    loss = fused.sum()
    loss.backward()
    
    # 检查梯度
    checks = [
        degraded.grad is not None and degraded.grad.abs().sum() > 0,
        all(n.grad is not None and n.grad.abs().sum() > 0 for n in normals),
    ]
    
    assert all(checks), "梯度回传失败！"
    print("  ✅ 梯度正常回传（端到端训练可行）")


def test_bandwidth_calculation():
    """测试带宽计算"""
    print("\n测试 4: 带宽使用计算")
    
    model = Who2comHandshake(feature_dim=256, message_dim=8, key_dim=1024)
    bw = model.get_bandwidth_usage(N_agents=5)
    
    print(f"  Who2com: {bw['ours_kb']:.1f} KB")
    print(f"  CatAll:  {bw['catall_kb']:.1f} KB")
    print(f"  节省比:  {1-bw['ratio']:.0%}")
    print("  ✅ 通过")


def test_empty_agents():
    """边界：没有正常agent"""
    print("\n测试 5: 无候选agent")
    
    model = Who2comHandshake(feature_dim=128)
    degraded = torch.randn(2, 128, 8, 8)
    
    weights, fused, selected = model(degraded, [], training=False)
    
    assert len(weights.flatten()) == 0
    assert fused.shape == degraded.shape
    print("  ✅ 通过")


if __name__ == "__main__":
    test_full_handshake_training()
    test_full_handshake_inference()
    test_gradient_through_handshake()
    test_bandwidth_calculation()
    test_empty_agents()
    print("\n" + "=" * 50)
    print("全部通过 ✅")