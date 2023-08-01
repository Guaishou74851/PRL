import torch
from torch import nn
import torch.nn.functional as F

PhiT_y, Phi_weight = None, None

class RB(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, padding=1),
        )
    
    def forward(self, x):
        return x + self.body(x)

class T(nn.Module):
    def __init__(self, nf, Phi_func, r, ID_nf):
        super().__init__()
        self.r = r
        self.body = nn.Sequential(
            nn.Conv2d(ID_nf, nf, 1),
            *[RB(nf) for _ in range(2)],
            nn.Conv2d(nf, ID_nf, 1),
        )
        self.Phi, self.PhiT = Phi_func
        self.G = nn.Sequential(
            nn.Conv2d(r*r+2*ID_nf, ID_nf, 3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x_input = x
        global PhiT_y
        x = F.pixel_shuffle(x, self.r)
        b, c, h, w = x.shape
        PhiT_Phi_x = self.PhiT(self.Phi(x.reshape(-1, 1, h, w))).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, self.r)
        x = x_input - self.G(torch.cat([x_input, PhiT_Phi_x, F.pixel_unshuffle(PhiT_y.to(x.device), self.r)], dim=1))
        return x + self.body(x)

class Operator(nn.Module):
    def __init__(self, nb, nf, Phi_func, r, ID_nf, mode=None):
        super().__init__()
        self.body = [T(nf, Phi_func, r, ID_nf) for _ in range(nb)]
        if mode == 'down':
            self.body.append(nn.Conv2d(ID_nf, ID_nf*4, 2, stride=2))
        elif mode == 'up':
            self.body = [nn.ConvTranspose2d(ID_nf*4, ID_nf, 2, stride=2)] + self.body
        else:
            self.body += [T(nf, Phi_func, r, ID_nf) for _ in range(nb)]
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        return self.body(x)
    
class UNet(nn.Module):
    def __init__(self, nb, Phi_func, nf, ID_nf):
        super().__init__()
        self.conv_first = nn.Conv2d(1, ID_nf, 3, padding=1)
        self.down1 = Operator(nb, nf, Phi_func, 1, ID_nf, 'down')
        self.down2 = Operator(nb, 4*nf, Phi_func, 2, ID_nf*4, 'down')
        self.mid = Operator(nb, 16*nf, Phi_func, 4, ID_nf*16)
        self.up2 = Operator(nb, 4*nf, Phi_func, 2, ID_nf*4, 'up')
        self.up1 = Operator(nb, nf, Phi_func, 1, ID_nf, 'up')
        self.conv_last = nn.Conv2d(ID_nf, 1, 3, padding=1)

    def forward(self, x):
        x0 = self.conv_first(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.mid(x2)
        x = self.up2(x)
        x = self.up1(x + x1)
        x = self.conv_last(x + x0)
        return x

class PRL(nn.Module):
    def __init__(self, nb, B, Phi_init, nf, ID_nf):
        super().__init__()
        self.ID_nf = ID_nf
        self.Phi_weight = nn.Parameter(Phi_init.view(-1, 1, B, B))
        self.Phi = lambda x: F.conv2d(x, Phi_weight.to(x.device), stride=B)
        self.PhiT = lambda x: F.conv_transpose2d(x, Phi_weight.to(x.device), stride=B)
        self.body = UNet(nb, [self.Phi, self.PhiT], nf, ID_nf)
        
    def forward(self, x):
        global PhiT_y, Phi_weight
        Phi_weight = self.Phi_weight.to(x.device)
        y = self.Phi(x)
        x = self.PhiT(y)
        PhiT_y = x.clone()
        x = self.body(x)
        return x
