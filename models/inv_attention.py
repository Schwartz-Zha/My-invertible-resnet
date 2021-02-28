import torch
import torch.nn as nn
from torch.nn import Parameter, Softmax

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
        B, C, H, W = x.size
        proj_query = self.query_conv(x).view(B, self.inter_c, -1, 1)  # [B, inter_c, HW, 1]
        proj_key = self.key_conv(x).view(B, self.inter_c, 1, -1)  # [B, inter_c, 1, HW]
        proj_query.repeat(1, 1, 1, H * W)
        proj_key.repeat(1, 1, H * W, 1)
        concat_feature = torch.cat([proj_query, proj_key], dim=1)  # [B, 2*inter_c, HW, HW]
        energy = self.concat_conv(concat_feature).squeeze()  # [B,  HW, HW]
        attention = energy / float(H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


def turbulance_hook(module, inputs):
    with torch.no_grad():
        res = module.forward(inputs)
        turbu_res = module.forward(inputs * 1.0000001)
        lip = torch.dist(turbu_res, res) / torch.dist(inputs, inputs * 1.0000001)
        if lip > 0.9:
            module.gamma = module.gamma * (0.9 / lip)
        else:
            pass


class InvAttention_concat(nn.Module):
    def __init__(self, in_c, k=4):
        super(InvAttention_concat, self).__init__()
        self.res_branch = Attention_concat(in_c, k=k)
        self.res_branch.register_forward_pre_hook(turbulance_hook)

