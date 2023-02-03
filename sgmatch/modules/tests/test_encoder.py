import torch

def test_mlpencoder():
    from sgmatch.modules.encoder import MLPEncoder

    x = torch.randn(4,16)
    mlp_layers = [32,16,32]
    mlp = MLPEncoder(node_feature_dim=x.shape[-1], node_hidden_sizes=mlp_layers)

    assert mlp.__repr__() == "MLPEncoder(node_hidden_sizes=[32, 16, 32], edge_hidden_sizes=None)"
    
    out = mlp(x)
    assert out.shape == torch.Size([4,32])

def test_orderembedder():
    pass