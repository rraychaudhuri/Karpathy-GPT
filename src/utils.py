import torch


default_device = "cpu"


def get_chunk_from_data(data, batch_size=4, block_size=8, encode=lambda x:x, d=default_device):
    """
    This function creates random chunks from the data which can be fed to the network
    """
    ix = torch.randint(len(data)-block_size-1, size=(batch_size,))
    X = torch.stack([torch.tensor(encode(data[i:i+block_size])) for i in ix.tolist()])
    Y = torch.stack([torch.tensor(encode(data[i+1:i+block_size+1])) for i in ix.tolist()])
    return X.to(d), Y.to(d)


@torch.no_grad()
def estimateLoss(model, data, batch_size=4, block_size=8, encode=lambda x:x, d=default_device, iter=200):
    model.eval()
    lossi = []
    for _ in range(iter):
        X, Y = get_chunk_from_data(data, batch_size, block_size, encode, d)
        _, loss = model(X, Y)
        lossi.append(loss.item())
    model.train()
    return sum(lossi)/len(lossi)

