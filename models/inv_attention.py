import torch
import torch.nn as nn
from torch.nn import Parameter, Softmax
from matrix_utils import power_series_matrix_logarithm_trace
from spectral_norm_fc import spectral_norm_fc

class Attention_concat(nn.Module):
    '''
      Concatenation Style PAM, with turbulance
    '''

    def __init__(self, in_c, k=4):
        super(Attention_concat, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // k
        self.query_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.concat_conv = nn.Conv2d(in_channels=self.inter_c * 2, out_channels=1, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, self.inter_c, -1, 1)  # [B, inter_c, HW, 1]
        proj_key = self.key_conv(x).view(B, self.inter_c, 1, -1)  # [B, inter_c, 1, HW]
        proj_query = proj_query.repeat(1, 1, 1, H * W)
        proj_key = proj_key.repeat(1, 1, H * W, 1)
        concat_feature = torch.cat([proj_query, proj_key], dim=1)  # [B, 2*inter_c, HW, HW]
        energy = self.concat_conv(concat_feature).squeeze()  # [B,  HW, HW]
        attention = energy / float(H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


def turbulance_hook(module, inputs):
    with torch.no_grad():
        x = inputs[0]
        res = module.forward(x)
        turbu_res = module.forward(x * 1.0000001)
        lip = torch.dist(turbu_res, res) / torch.dist(x, x * 1.0000001)
        if lip > 0.9:
            module.gamma *=(0.9 / lip)
        else:
            pass


class InvAttention_concat(nn.Module):
    def __init__(self, in_c, k=4, numTraceSamples = 1, numSeriesTerms = 5):
        super(InvAttention_concat, self).__init__()
        self.res_branch = Attention_concat(in_c, k=k)
        self.res_branch.register_forward_pre_hook(turbulance_hook)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x

class Attention_dot(nn.Module):
    '''
    Dot product, Softmax
    '''
    def __init__(self, input_channel_num, k=4):
        super(Attention_dot, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, attention).view(B, C, H, W)

        out = torch.clamp(self.gamma, min=-1.0, max=1.0) * out + x
        return out

class InvAttention_dot(nn.Module):
    def __init__(self, input_channel_num, k=4, numTraceSamples=1, numSeriesTerms=5):
        super(InvAttention_dot, self).__init__()
        self.res_branch= Attention_dot(input_channel_num, k=k)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x

class Attention_dot2(nn.Module):
    '''
    Dot product, inv
    '''
    def __init__(self, input_channel_num, k=4):
        super(Attention_dot2, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.nonlin = nn.ELU()
        self.gamma = Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        energy = self.nonlin(energy)
        with torch.no_grad():
            energy_sum = torch.sum(energy,dim=(1,2), keepdim=True)
        energy = energy / (1.5 * energy_sum) #hooray
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        out = torch.clamp(self.gamma, min=-1.0, max=1.0) * out + x
        return out

class InvAttention_dot2(nn.Module):
    def __init__(self, input_channel_num, k=4, numTraceSamples=1, numSeriesTerms=5):
        super(InvAttention_dot2, self).__init__()
        self.res_branch= Attention_dot2(input_channel_num, k=k)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x


class Attention_dot3(nn.Module):
    '''
    Dot product, inv
    '''
    def __init__(self, input_channel_num, k=4):
        super(Attention_dot3, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.nonlin = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        energy = self.nonlin(energy)
        energy = energy / float(H*W * H*W)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        out = torch.clamp(self.gamma, min=-1.0, max=1.0) * out + x
        return out

class InvAttention_dot3(nn.Module):
    def __init__(self, input_channel_num, k=4, numTraceSamples=1, numSeriesTerms=5):
        super(InvAttention_dot3, self).__init__()
        self.res_branch= Attention_dot3(input_channel_num, k=k)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x