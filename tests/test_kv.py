from types import SimpleNamespace

from llmbench.runner import kv_cache_bytes


def test_kv_cache_mha():
    cfg = SimpleNamespace(num_hidden_layers=2, num_attention_heads=4, hidden_size=64)
    # 2 * layers(2) * heads(4) * head_dim(16) * tokens(10) * batch(1) * 2 bytes
    assert kv_cache_bytes(cfg, 1, 10) == 2 * 2 * 4 * 16 * 10 * 1 * 2


def test_kv_cache_gqa_is_smaller():
    mha = SimpleNamespace(num_hidden_layers=2, num_attention_heads=8, hidden_size=64)
    gqa = SimpleNamespace(num_hidden_layers=2, num_attention_heads=8, hidden_size=64, num_key_value_heads=2)
    assert kv_cache_bytes(gqa, 1, 100) * 4 == kv_cache_bytes(mha, 1, 100)
