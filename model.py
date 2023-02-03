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

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.num_labels = num_labels

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        # print('VAE forward')
        view_size = 1000
        if x.dim() > 2:
            x = x.view(-1, view_size)

        batch_size = x.size(0)
        
        if x.is_cuda != True:
            x = x.cuda()
        if c.is_cuda != True:
            c = c.cuda()
        # print(f'in VAE forward, x cuda? : {x.is_cuda} {x.dtype}')

        means, log_var = self.encoder(x, c)
        # print(f'in VAE forward, means cuda? : {means.is_cuda} {means.dtype}')
        # print(f'in VAE forward, log_var cuda? : {log_var.is_cuda} {log_var.dtype}')

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        if eps.is_cuda != True:
            eps = eps.cuda()
        # print(f'in VAE forward, std cuda? : {std.is_cuda} {std.dtype}')
        # print(f'in VAE forward, eps cuda? : {eps.is_cuda} {eps.dtype}')
        z = eps * std + means
        
        # print(f'in VAE forward, z cuda? : {z.is_cuda} {z.dtype}')
        # print(f'in VAE forward, c cuda? : {c.is_cuda} {c.dtype}')
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
        #if x.dim() > 2:
        #    x = x.view(-1, view_size)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([1, self.latent_size])
        
        z = eps * std + means

        return z


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        # print('E init')

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        self.num_labels = num_labels

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # print(f'layer_sizes의 (in_size, out_size) : {in_size, out_size}')
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        # print('E forward')
        # print(f'in E forward, x cuda? : {x.is_cuda} {x.dtype}')
        # print(f'in E forward, c cuda? : {c.is_cuda} {c.dtype}')
        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        # print(f'in E forward, x cuda? : {x.is_cuda}')
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        # print(f'in E forward, means cuda? : {means.is_cuda} {means.dtype}')
        # print(f'in E forward, log_vars cuda? : {log_vars.is_cuda} {log_vars.dtype}')

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        # print('D init')

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
        # print('D forward')
        if self.conditional:
            # print(f'in D forward, c dtype? : {c.dtype} {type(c)}')
            # print(f'in D forward, z dtype? : {z.dtype} {type(z)}')
            if type(c) != torch.Tensor:
                c = torch.from_numpy(c)
            c = c.cuda()
            z = z.cuda()
            # print(f'in D forward, c cuda? : {c.is_cuda} {c.dtype}')
            # print(f'in D forward, z cuda? : {z.is_cuda} {z.dtype}')
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x