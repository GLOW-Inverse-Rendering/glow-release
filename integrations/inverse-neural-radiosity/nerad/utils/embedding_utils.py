import torch
from nerad.model.tcnn_embedding import TcnnEmbedding


def create_embedding(config, num_input_dim=3):
    if num_input_dim != 3:
        assert config['otype'] == "HashGrid", config['otype']
    if config['otype'] == 'SparseGrid':
        not NotImplementedError()
        embedding = MutliResGrid(**config)
    elif config['otype'] == 'DenseGrid':
        not NotImplementedError()
        embedding = MutliResGrid(**config)
    else:
        embedding = TcnnEmbedding(config, num_input_dim=num_input_dim)
    return embedding

def embed(input_, embedding):
    embed_type = embedding.embedding_type
    net_in = None
    match embed_type:
        case "identity":
            net_in = input_
        case "SparseGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "DenseGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "HashGrid":
            net_in = torch.cat([input_, embedding(input_)], dim=-1)
        case "Grid":
            net_in = torch.cat([input_, embedding((input_-0.5))], dim=-1)
        case "Frequency":
            net_in = embedding(2*input_-1)
        case "SphericalHarmonics":
            net_in = embedding(input_)
        case _:
            raise Exception("Unhandled embedding")
    return net_in
