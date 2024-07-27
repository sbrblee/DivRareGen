import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
        with torch.no_grad():
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
        
            # Apply initialization parameters
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, n_F, _ = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        
        log_abs = torch.log(torch.abs(self.scale))

        logdet = torch.sum(log_abs) * n_F

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class InvConv1d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        input = input.reshape((input.shape[0], input.shape[1], input.shape[2]))
        _, n_channel, n_F = input.shape

        out = F.conv1d(input, self.weight)
        out = out.unsqueeze(-1)
        logdet = (
            n_F * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        output = output.reshape((output.shape[0], output.shape[1], output.shape[2]))
        out = F.conv1d(
            output, self.weight.squeeze().inverse().unsqueeze(2)
        )
        out = out.unsqueeze(-1)
        return out


class InvConv1dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        input = input.reshape((input.shape[0], input.shape[1], input.shape[2]))
        _, n_channel, n_F = input.shape

        weight = self.calc_weight()

        out = F.conv1d(input, weight)
        logdet = n_F * torch.sum(self.w_s)

        out = out.unsqueeze(-1)
        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2)

    def reverse(self, output):
        output = output.reshape((output.shape[0], output.shape[1], output.shape[2]))
        weight = self.calc_weight()
        
        out = F.conv1d(output, weight.squeeze().inverse().unsqueeze(-1))
        out = out.unsqueeze(-1)
        return out

class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        # Assuming in_channel is 4096 and out_channel is the desired output size
        self.conv = nn.Conv1d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        input = input.reshape((input.shape[0], input.shape[1], input.shape[2]))
        out = F.pad(input, [1, 1], value=1)
        out = self.conv(out)
        out = out.unsqueeze(-1)
        out = out * torch.exp(self.scale * 3)

        return out
        

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv1d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        b_size, _, _, _ = input.shape
        input = input.reshape((input.shape[0], input.shape[1], input.shape[2]))

        in_a, in_b = input.chunk(2, 1)
        
        in_b = in_b.unsqueeze(-1)

        if self.affine:
            # Split the output of the network into log_s and t
            log_s, t = self.net(in_a).chunk(2, 1)
            # Apply affine transformation
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            # Compute log determinant
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            # Apply coupling transformation
            out_b = in_b + self.net(in_a)
            logdet = None

        in_a = in_a.unsqueeze(-1)
        # Concatenate transformed parts
        return torch.cat([in_a, out_b], dim=1), logdet

    def reverse(self, output):
        b_size, _, _, _ = output.shape

        # Reshape output to [batch_size, 4096] for linear transformation
        output = output.reshape((output.shape[0], output.shape[1], output.shape[2]))

        out_a, out_b = output.chunk(2, 1)
        
        out_b = out_b.unsqueeze(-1)

        if self.affine:
            # Split the output of the network into log_s and t
            log_s, t = torch.chunk(self.net(out_a), 2, 1)
            # Apply inverse affine transformation
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            # Apply inverse coupling transformation
            in_b = out_b - self.net(out_a)

        out_a = out_a.unsqueeze(-1)
        # Concatenate parts in reverse
        return torch.cat([out_a, in_b], dim=1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        
        self.actnorm = ActNorm(in_channel)
        
        if conv_lu:
            self.invconv = InvConv1dLU(in_channel)

        else:
            self.invconv = InvConv1d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        b_size, _, _, _ = input.shape

        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        b_size, _, _, _ = output.shape

        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        
        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, group=4, n_F=4096):
        super().__init__()

        squeeze_dim = in_channel * group

        self.flows = nn.ModuleList()
        if conv_lu:
            self.initial_permute = InvConv1dLU(n_F)
        else:
            self.initial_permute = InvConv1d(n_F)
        
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv1d(in_channel * (group//2), in_channel * group)
        else:
            self.prior = ZeroConv1d(in_channel * group, in_channel * (group*2))
            
        self.group = group

    def forward(self, input):
        b_size, n_F, n_channel, _ = input.shape

        out, logdet = self.initial_permute(input)
        
        out = out.permute(0, 2, 1, 3)  # Swap dimensions for Conv1d compatibility
        out = out.contiguous().view(b_size, n_channel*self.group, n_F//self.group, 1)

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, n_F, _ = input.shape

        input = input.contiguous().view(b_size, n_channel//self.group, n_F*self.group, 1)
        input = input.permute(0, 2, 1, 3)
        
        out = self.initial_permute.reverse(input)
        
        return out


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True, group=4, n_F=4096
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        n_F_ = n_F
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, group=group, n_F=n_F_))
            n_channel *= group//2
            n_F_ //= group
            
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, n_F=n_F_, group=group))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            out = out.permute(0,2,1,3)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
            

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                # Swap dimensions back to original before reversing
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct).permute(0,2,1,3)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct).permute(0,2,1,3)

        return input
