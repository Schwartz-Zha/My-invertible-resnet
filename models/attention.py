import torch
import torch.nn as nn
from torch.nn import Parameter, Softmax

class PAM_Module(nn.Module):
    """ Position attention module"""

    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # [B, HW, C]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # [B, C, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        attention = self.softmax(energy)  # [B, HW, HW]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # [B, C, HW]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # batch matrix multiplication,
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_Module_v2(nn.Module):
    '''
    Little Bit Simplified Positional Attention Module
    '''

    def __init__(self, input_channel_num):
        super(PAM_Module_v2, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1)
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
        out = self.gamma * out + x
        return out


class PAM_Module_v3(nn.Module):
    '''
    Dot Product Positional Attention Module
    '''

    def __init__(self, in_c):
        super(PAM_Module_v3, self).__init__()
        self.in_c = in_c
        self.query_conv = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = energy / float(H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, attention).view(B, C, H, W)
        out = self.gamma * out + x
        return out


class PAM_Module_v4(nn.Module):
    '''
    Concatenation Style PAM
    '''

    def __init__(self, in_c):
        super(PAM_Module_v4, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // 8
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


class PAM_Module_v5(nn.Module):
    '''
    Deepmind proposed attention with Lipschitz constant
    '''

    def __init__(self, in_c):
        super(PAM_Module_v5, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // 8
        self.query_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.value_conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, H * W, -1)  # [B, HW, inter_c]
        proj_key = self.key_conv(x).view(B, H * W, -1)  # [B, HW, inter_c]
        energy = -torch.cdist(proj_query, proj_key) / float(H * W)  # [B, HW, HW]
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


class PAM_Module_v6(nn.Module):
    '''
        Concatenation Style PAM, with non-lin bound
    '''

    def __init__(self, in_c):
        super(PAM_Module_v6, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // 8
        self.query_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_c, out_channels=self.inter_c, kernel_size=1)
        self.concat_conv = nn.Conv2d(in_channels=self.inter_c * 2, out_channels=1, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.nonlin = nn.Tanh()

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
        proj_value = self.nonlin(proj_value)
        out = torch.bmm(proj_value, attention).view(B, -1, H, W)
        out = self.gamma * out + x
        return out


class PAM_Module_v7:
    '''
        Concatenation Style PAM, with turbulance
    '''

    def __init__(self, in_c):
        super(PAM_Module_v7, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // 8
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


if __name__ == '__main__':
    demo_input = torch.randn(32, 12, 16, 16)
    layer = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
    layer.register_forward_pre_hook(turbulance_hook)
    layer.forward(demo_input)
