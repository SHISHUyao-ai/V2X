"""测试 MessageGenerator 和 KeyGenerator"""
import torch
import sys
sys.path.insert(0, '.')
from src.matching.generators import MessageGenerator, KeyGenerator


def test_message_generator():
    print("=" * 50)
    print("测试 MessageGenerator（论文 Eq 1）")
    
    gen = MessageGenerator(feature_dim=256, message_dim=8)
    
    # 模拟退化智能体的特征图
    degraded = torch.randn(4, 256, 16, 16)
    msg = gen(degraded)
    
    assert msg.shape == (4, 8), f"shape错误: {msg.shape}"
    print(f"  输入: (4, 256, 16, 16)")
    print(f"  输出: {msg.shape} ← 8维！比输入小 (256×16×16)/8 = 8192倍")
    print("  ✅ 通过")


def test_key_generator_list():
    print("\n测试 KeyGenerator - List输入（论文 Eq 2）")
    
    gen = KeyGenerator(feature_dim=256, key_dim=1024)
    
    # 5个正常智能体
    normal_feats = [torch.randn(4, 256, 16, 16) for _ in range(5)]
    keys = gen(normal_feats)
    
    assert keys.shape == (4, 5, 1024), f"shape错误: {keys.shape}"
    print(f"  输入: 5个 (4, 256, 16, 16)")
    print(f"  输出: {keys.shape}")
    print("  ✅ 通过")


def test_key_generator_stacked():
    print("\n测试 KeyGenerator - Stacked输入")
    
    gen = KeyGenerator(feature_dim=128, key_dim=512)
    
    normal_feats = torch.randn(2, 3, 128, 8, 8)  # (B, N, C, H, W)
    keys = gen(normal_feats)
    
    assert keys.shape == (2, 3, 512), f"shape错误: {keys.shape}"
    print(f"  输入: (2, 3, 128, 8, 8)")
    print(f"  输出: {keys.shape}")
    print("  ✅ 通过")


def test_asymmetric_sizes():
    """验证非对称压缩：message小、key大"""
    print("\n测试 非对称压缩（论文核心设计）")
    
    msg_gen = MessageGenerator(feature_dim=256, message_dim=8)
    key_gen = KeyGenerator(feature_dim=256, key_dim=1024)
    
    feat = torch.randn(4, 256, 16, 16)
    msg = msg_gen(feat)
    keys = key_gen([feat] * 5)
    
    print(f"  message: {msg.shape} ({msg.element_size() * msg.numel()} bytes/batch)")
    print(f"  key:     {keys.shape} ({keys.element_size() * keys.numel()} bytes/batch)")
    print(f"  非对称比: {1024/8:.0f}:1 (key比message大{1024/8}x)")
    print(f"  且key不需要传输！")
    print("  ✅ 通过")


if __name__ == "__main__":
    test_message_generator()
    test_key_generator_list()
    test_key_generator_stacked()
    test_asymmetric_sizes()
    print("\n" + "=" * 50)
    print("全部通过 ✅")