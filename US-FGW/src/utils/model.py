"""
introduction to models 
--------------------------------------------------------------------^
dim of input: [I,64] or [J,48/16/80]
encoder-decoder structure based on MLP:
AE_video_MLP                     {encoder: 64->32->16->z_dim, decoder: z_dim->16->32->64}
AE_word_breakfast_MLP            {encoder: 48->32->16->z_dim, decoder: z_dim->16->32->48}
AE_word_hollywood_MLP            {encoder: 16->10->z_dim, decoder: z_dim->10->16}
AE_word_crosstask_MLP            {encoder: 80->40->16->z_dim, decoder: z_dim->16->40->80}
---------------------------------------------------------------------v
"""
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import random 


def key_frame(M, sample_rate=10, device='cuda'):
    '''
    extract key frames every 10 frames randomly. Specially, The last key frame is extracted from last I%10 frames. 
    '''
    r = M.size(0) // sample_rate + (M.size(0) % sample_rate != 0) # number of rows of M (i.e. sequence.T)
    tmp = torch.zeros(r, M.size(1)).to(device)
    for i in range(M.size(0) // sample_rate):
        j = random.randint(0, sample_rate - 1)
        tmp[i , :] = M[sample_rate * i + j , :]
    if M.size(0) % sample_rate == 0:
        # tmp[-1 , :] = M[sample_rate*(i + 1) - 1]
        pass
    else:
        tmp[-1 , :] = M[sample_rate * (i + 1) + random.randint(0 , M.size(0) % sample_rate - 1)]
    return tmp

def loss_function(recon_x, x, rec_type, transcript=None, device=None):
    if rec_type == 'BCE':
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif rec_type == 'MSE':
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')
    elif rec_type == 'MAE':
        reconstruction_loss = F.l1_loss(recon_x, x, reduction='sum')
    elif rec_type == 'CrossEntropy':
        #print(x.size())
        transcript = torch.tensor(transcript)
        transcript = transcript.to(device)
        criteria = nn.CrossEntropyLoss(reduction='mean')
        reconstruction_loss =criteria(recon_x, transcript)
    else:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') + F.l1_loss(recon_x, x, reduction='sum')
    return reconstruction_loss


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE_video_MLP(nn.Module):
    """MLP Encoder-Decoder architecture for video."""
    def __init__(self, z_dim=8, model_type='probabilistic', sample_rate=10, device='cpu'):
        super(AE_video_MLP, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        self.sample_rate = sample_rate
        self.device = device
        self.gru = nn.GRU(64, 64, 1, 
                            bidirectional = False, batch_first = True)
        # [I,64]
        self.encoder = nn.Sequential(
            nn.Linear(64,32), # [I,32]
            nn.ReLU(True),
            nn.Linear(32,16), # [I,16]
        )

        if self.type == 'probabilistic':
            self.fc1 = nn.Linear(16, z_dim)    # [I,z_dim]
            self.fc2 = nn.Linear(16, z_dim)    # [I,z_dim]
        else:
            self.fc = nn.Linear(16, z_dim)    # [I,z_dim]

        self.decoder = nn.Sequential(
            nn.Linear(z_dim,16), # [I,16]
            nn.ReLU(True),
            nn.Linear(16,32), # [I,32]
            nn.ReLU(True),
            nn.Linear(32,64), # [I,64]
    )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.gru(x)
        x = x[0][:, -1, :] # L * hidden_size

        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x) 
            x_recon = self.decode(z) # [I,64]
            self.rec_loss = loss_function(x_recon, x, 'MSE', None, None)
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x) # [I,z_dim]
            x_recon = self.decode(z)
            self.rec_loss = loss_function(x_recon, x, 'MSE', None, None)
            return x_recon, z

    def encode(self, x):
        z = self.encoder(x) # [I,16]
        if self.type == 'probabilistic':
            mu = self.fc1(z) # [I,z_dim]
            logvar = self.fc2(z) # [I,z_dim]
            z = self.reparameterize(mu, logvar) # [I,z_dim]
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z # [I,z_dim]

    def decode(self, z):
        return self.decoder(z)


class AE_word_breakfast_MLP(nn.Module):
    """MLP Encoder-Decoder architecture for word_breakfast."""
    def __init__(self, z_dim=8,model_type='probabilistic'):
        super(AE_word_breakfast_MLP, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        # [J,48]
        self.encoder = nn.Sequential(
            nn.Linear(48,32), # [J,32]
            nn.ReLU(True),
            nn.Linear(32,16), # [J,16]
        )

        if self.type == 'probabilistic':
            self.fc1 = nn.Linear(16, z_dim)    # [J,z_dim]
            self.fc2 = nn.Linear(16, z_dim)    # [J,z_dim]
        else:
            self.fc = nn.Linear(16, z_dim)    # [J,z_dim]

        self.decoder = nn.Sequential(
            nn.Linear(z_dim,16), # [J,16]
            nn.ReLU(True),
            nn.Linear(16,32), # [J,32]
            nn.ReLU(True),
            nn.Linear(32,48), # [J,48]
    )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x) 
            x_recon = self.decode(z) # [J,48]
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    def encode(self, x):
        z = self.encoder(x) # [J,16]
        if self.type == 'probabilistic':
            mu = self.fc1(z) # [J,z_dim]
            logvar = self.fc2(z) # [J,z_dim]
            z = self.reparameterize(mu, logvar) # [J,z_dim]
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z # [J,z_dim]

    def decode(self, z):
        return self.decoder(z)

class AE_word_hollywood_MLP(nn.Module):
    """MLP Encoder-Decoder architecture for word_hollywood."""
    def __init__(self, z_dim=8, model_type='probabilistic'):
        super(AE_word_hollywood_MLP, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        # [I,16]
        self.fc1 = nn.Linear(16, 10)
        if self.type == 'probabilistic':
            self.fc21 = nn.Linear(10, z_dim)
            self.fc22 = nn.Linear(10, z_dim)
        else:
            self.fc2 = nn.Linear(10, z_dim)
        self.fc3 = nn.Linear(z_dim, 10)
        self.fc4 = nn.Linear(10, 16)

    def encode(self, x):
        z = F.relu(self.fc1(x)) # [J,10]
        if self.type == 'probabilistic':
            mu = self.fc21(z) # [J,z_dim]
            logvar = self.fc22(z) # [J,z_dim]
            z = self.reparameterize(mu, logvar) # [J,z_dim]
            return z, mu, logvar
        else:
            z = self.fc2(z) # [J,z_dim]
            return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

class AE_word_crosstask_MLP(nn.Module):
    """MLP Encoder-Decoder architecture for word_crosstask."""
    def __init__(self, z_dim=8,model_type='probabilistic'):
        super(AE_word_crosstask_MLP, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        # [J,80]
        self.encoder = nn.Sequential(
            nn.Linear(80,40), # [J,40]
            nn.ReLU(True),
            nn.Linear(40,16), # [J,16]
        )

        if self.type == 'probabilistic':
            self.fc1 = nn.Linear(16, z_dim)    # [J,z_dim]
            self.fc2 = nn.Linear(16, z_dim)    # [J,z_dim]
        else:
            self.fc = nn.Linear(16, z_dim)    # [J,z_dim]

        self.decoder = nn.Sequential(
            nn.Linear(z_dim,16), # [J,16]
            nn.ReLU(True),
            nn.Linear(16,40), # [J,40]
            nn.ReLU(True),
            nn.Linear(40,80), # [J,80]
    )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x) 
            x_recon = self.decode(z) # [J,80]
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    def encode(self, x):
        z = self.encoder(x) # [J,16]
        if self.type == 'probabilistic':
            mu = self.fc1(z) # [J,z_dim]
            logvar = self.fc2(z) # [J,z_dim]
            z = self.reparameterize(mu, logvar) # [J,z_dim]
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z # [J,z_dim]

    def decode(self, z):
        return self.decoder(z)

