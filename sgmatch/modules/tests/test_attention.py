import torch

def test_globalcontextatt():
    from sgmatch.modules.attention import GlobalContextAttention

    x = torch.randn(4, 16)

    att = GlobalContextAttention(input_dim=x.shape[1], activation="tanh")
    assert att.__repr__() == "GlobalContextAttention(input_dim=16)"
    out = att(x)
    assert out.shape == torch.Size([16])

def test_crossgraphatt():
    from sgmatch.modules.attention import CrossGraphAttention

    x_i = torch.randn(4, 16)
    x_j = torch.randn(5,16)

    att = CrossGraphAttention()
    assert att.__repr__() == "CrossGraphAttention()"
    out_i, out_j = att(x_i, x_j)

    assert out_i.shape == x_i.shape
    assert out_j.shape == x_j.shape
    