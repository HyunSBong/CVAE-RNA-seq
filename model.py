import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def idx2onehot(idx, n):
 
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1) # 차원이 1인 차원을 제거
    idx = idx.cuda()
    onehot = torch.zeros(idx.size(0), n).cuda() # 128,15
    onehot = onehot.scatter(1, idx, 1).cuda() # scatter(dim,index,src,value) dim은 axis(1:(가로)열방향)
    """
    tensor([[0.0000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000]])
    """

    return onehot

class VAE(nn.Module):

    def __init__(self, embedding_dim, latent_size, decompress_dims,
                 conditional=True, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(embedding_dim) == list
        assert type(latent_size) == int
        assert type(decompress_dims) == list

        self.latent_size = latent_size
        self.num_labels = num_labels

        self.encoder = Encoder(
            embedding_dim, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decompress_dims, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        view_size = 1000
        if x.dim() > 2:
            x = x.view(-1, view_size)

        batch_size = x.size(0)
        
        if x.is_cuda != True:
            x = x.cuda()
        if c.is_cuda != True:
            c = c.cuda()

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        if eps.is_cuda != True:
            eps = eps.cuda()
        z = eps * std + means
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=0, c=None):
        if n == 0:
            n = self.num_labels
        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x

    def embedding(self, x, c=None):
        view_size = 1000

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([1, self.latent_size])
        
        z = eps * std + means

        return z

class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        self.num_labels = num_labels

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()
        self.num_labels = num_labels

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        if self.conditional:
            if type(c) != torch.Tensor:
                c = torch.from_numpy(c)
            c = c.cuda()
            z = z.cuda()
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x