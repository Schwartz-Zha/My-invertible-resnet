"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from .model_utils import injective_pad, ActNorm2D, Split
from .model_utils import squeeze as Squeeze
from .model_utils import MaxMinGroup
from spectral_norm_conv_inplace import spectral_norm_conv
from spectral_norm_fc import spectral_norm_fc
from matrix_utils import power_series_matrix_logarithm_trace
from .inv_attention import InvAttention_concat, InvAttention_dot, InvAttention_gaussian, InvAttention_embedded_gaussian


# class LogisticTransform(torch.distributions.Transform):
#     r"""
#     Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
#     """
#     codomain = constraints.real
#     domain = constraints.unit_interval
#     bijective = True
#     sign = +1
#
#     def __eq__(self, other):
#         return isinstance(other, LogisticTransform)
#
#     def _call(self, x):
#         return x.log() - (-x).log1p()
#
#     def _inverse(self, y):
#         return torch.sigmoid(y)
#
#     def log_abs_det_jacobian(self, x, y):
#         return F.softplus(y) + F.softplus(-y)


# def logistic_distribution(loc, log_scale):
#     scale = torch.exp(log_scale) + 1e-5
#     base_distribution = distributions.Uniform(torch.zeros_like(loc), torch.ones_like(loc))
#     transforms = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
#     logistic = distributions.TransformedDistribution(base_distribution, transforms)
#     return logistic


def downsample_shape(shape):
    return (shape[0] * 4, shape[1] // 2, shape[2] // 2)


class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size2 = 1 # kernel size for second conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0),
                                                  (int_ch, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, trace + an_logdet

    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.bottleneck_block(x)
                x = y - summand

            if self.actnorm is not None:
                x = self.actnorm.inverse(x)

            # inversion of squeeze (dimension shuffle)
            if self.stride == 2:
                x = self.squeeze.inverse(x)
            return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)


class scale_block(nn.Module):
    def __init__(self, steps, in_shape, int_dim, squeeze=True, do_attention=False, n_terms=0, n_samples=0,
                 coeff=.9, input_nonlin=True, actnorm=True, split=True,
                 n_power_iter=5, nonlin="relu"):
        super(scale_block, self).__init__()
        self.in_shape = in_shape
        if squeeze:
            self.squeeze = Squeeze(2)
            conv_shape = downsample_shape(in_shape)
        else:
            self.squeeze = None
            conv_shape = in_shape

        if split:
            self.split = Split()
            n = int(conv_shape[0] // 2)
            out_shape1 = (n, conv_shape[1], conv_shape[2])
            out_shape2 = (conv_shape[0] - n, conv_shape[1], conv_shape[2])
            self.out_shapes = [out_shape1, out_shape2]
        else:
            self.split = None
            self.out_shapes = [conv_shape]

        self.stack = self._make_stack(steps, n_terms, n_samples, conv_shape, int_dim,
                                      input_nonlin, coeff, actnorm, n_power_iter, nonlin)
        if do_attention:
            #self.attention = InvAttention_concat(conv_shape[0], k=4, numTraceSamples=n_samples, numSeriesTerms=n_terms,
            #                                     convGamma=True)
            # self.attention = InvAttention_gaussian(conv_shape[0]  , numTraceSamples=n_samples, numSeriesTerms=n_terms,
            #                                      convGamma=True)
            # self.attention = InvAttention_embedded_gaussian(conv_shape[0], k=4, numTraceSamples=n_samples, numSeriesTerms=n_terms,
            #                                      convGamma=True)
            self.attention = InvAttention_dot(conv_shape[0], k=4, numTraceSamples=n_samples, numSeriesTerms=n_terms,
                                                  convGamma=True)
        else:
            self.attention = None

    @staticmethod
    def _make_stack(steps, n_terms, n_samples, in_shape, int_dim,
                    input_nonlin, coeff, actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        for i in range(steps):
            block_list.append(conv_iresnet_block(in_shape, int_dim, n_samples, n_terms,
                                                 stride=1, input_nonlin=True if input_nonlin else i > 0,
                                                 coeff=coeff, actnorm=actnorm,
                                                 n_power_iter=n_power_iter, nonlin=nonlin))

        return block_list

    def forward(self, x, ignore_logdet=False):
        if self.squeeze is not None:
            x = self.squeeze(x)

        traces = []
        z = x
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)

        trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            trace += traces[k]

        if self.attention is not None:
            z, tmp_trace = self.attention(z, ignore_logdet=ignore_logdet)
            trace = trace + tmp_trace

        if self.split is None:
            return [z], trace
        else:
            z1, z2 = self.split(z)
            return [z1, z2], trace

    def inverse(self, z, z2=None, maxIter=100):
        with torch.no_grad():
            if self.split is None:
                x = z
            else:
                assert z2 is not None
                x = self.split.inverse(z, z2)

            if self.attention is not None:
                x = self.attention.inverse(x)

            for block in reversed(self.stack):
                x = block.inverse(x, maxIter=maxIter)

            if self.squeeze is None:
                return x
            else:
                return self.squeeze.inverse(x)


class multiscale_conv_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides,nChannels, doAttention,init_squeeze=False,
                 coeff=.9, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 actnorm=True, nonlin="relu", use_label=True):
        super(multiscale_conv_iResNet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
        if init_squeeze:
            self.init_squeeze = Squeeze(2)
        else:
            self.init_squeeze = None

        self.input_shape = in_shape

        if init_squeeze:
            in_shape = downsample_shape(in_shape)
        in_shape = (in_shape[0], in_shape[1], in_shape[2])  # adjust channels

        self.nBlocks = nBlocks
        self.nClasses = nClasses
        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter

        self.stack, self.in_shapes = self._make_stack(in_shape, nBlocks,
                                                      nStrides, nChannels, doAttention, numSeriesTerms, numTraceSamples,
                                                      coeff, actnorm, n_power_iter, nonlin)
        # make prior distribution
        if use_label:
            self._make_prior_label()
        else:
            self._make_prior()
        self.use_label = use_label

    def final_shape(self):
        return self.stack[-1].out_shapes[-1]

    def z_shapes(self):
        shapes = []
        for block in self.stack:
            if len(block.out_shapes) == 2:
                shapes.append(block.out_shapes[0])
        shapes.append(self.final_shape())
        return shapes

    def get_in_shapes(self):
        return self.in_shapes

    def _make_stack(self, in_shape, nSteps, nStrides, nChannels, doAttention, n_terms,
                    n_samples, coeff, actnorm, n_power_iter, nonlin):
        blocks = nn.ModuleList()
        n_blocks = len(nSteps)
        in_shapes = [in_shape]
        for i, (steps, stride, channels, do_attention) in enumerate(zip(nSteps, nStrides, nChannels, doAttention)):
            block = scale_block(steps, in_shape, channels,
                                stride == 2, do_attention, n_terms, n_samples,
                                coeff, i > 0, actnorm,
                                i < n_blocks - 1,
                                n_power_iter, 
                                nonlin)  # split on all but last layer
            in_shape = block.out_shapes[-1]
            in_shapes.append(in_shape)
            blocks.append(block)
        return blocks, in_shapes

    def _make_prior(self):
        dim = torch.prod(torch.tensor(self.in_shapes[0]))
        self.prior_mu = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=True)
        self.prior_logstd = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=True)

    def _make_prior_label(self):
        self.mean_net = LabelNet(self.nClasses, self.input_shape)
        self.logstd_net = LabelNet(self.nClasses, self.input_shape)

    def prior(self):
        return distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))

    def logpz(self, z):
        return self.prior().log_prob(z.view(z.size(0), -1)).sum(dim=1)

    def logpz_label(self, z, labels, nClasses=10):
        labels = labels.view(z.size(0), 1)
        labels_onehot = torch.zeros(labels.size(0), nClasses).to(labels.device)
        labels_onehot.scatter_(1, labels, 1)
        mean = self.mean_net(labels_onehot).view(z.size(0), -1)
        logstd = self.logstd_net(labels_onehot).view(z.size(0), -1)
        return torch.distributions.Normal(mean, torch.exp(logstd)).log_prob(z).sum(dim=1)

    def forward(self, x, labels ,ignore_logdet=False):
        """ iresnet forward """
        if self.init_squeeze is not None:
            x = self.init_squeeze.forward(x)

        zs = []
        traces = []
        cur_act = x
        for block in self.stack:
            this_zs, trace = block(cur_act, ignore_logdet=ignore_logdet)
            if len(this_zs) == 1:
                cur_act = this_zs[0]
            else:
                assert len(this_zs) == 2
                cur_act = this_zs[1]
                zs.append(this_zs[0])
            traces.append(trace)
        zs.append(cur_act)  # add last activation to zs

        # add logdets
        tmp_trace = torch.zeros_like(traces[0])
        for k in range(len(traces)):
            tmp_trace += traces[k]

        bs = zs[0].size(0)
        zs_flat = [z.view(bs, -1) for z in zs]
        z = torch.cat(zs_flat, 1)
        if self.use_label:
            logpz = self.logpz_label(z, labels, self.nClasses)
        else:
            logpz = self.logpz(z)
        return zs, logpz, tmp_trace


    def inverse(self, zs, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            cur_act = zs[-1]
            zs = zs[:-1]
            for block in reversed(self.stack):
                # if this block has a split
                if len(block.out_shapes) == 2:
                    # pass in top z and cur act
                    cur_act = block.inverse(zs[-1], cur_act, maxIter=max_iter)
                    # shorten zs
                    zs = zs[:-1]
                # if there is no split
                else:
                    cur_act = block.inverse(cur_act, maxIter=max_iter)

            x = cur_act

            if self.init_squeeze is not None:
                x = self.init_squeeze.inverse(x)
        return x

    def split_zs(self, z):
        zs = []
        cur_dim = 0
        for z_shape in self.z_shapes():
            z_dim = torch.prod(torch.tensor(z_shape))
            this_z = z[:, cur_dim: cur_dim + z_dim]
            this_z = this_z.view(z.size(0), *z_shape)
            zs.append(this_z)
            cur_dim += z_dim
        return zs


    def sample(self, batch_size, max_iter=100):
        """sample from prior and invert"""
        with torch.no_grad():
            if self.use_label:
                samples_list = []
                for i in range(self.nClasses):
                    pseudo_label = torch.zeros([1, self.nClasses])
                    pseudo_label[0, i] = 1
                    pseudo_label = pseudo_label.to(next(self.mean_net.parameters()).device)
                    mean = self.mean_net(pseudo_label).view(-1)
                    logstd = self.logstd_net(pseudo_label).view(-1)
                    dist = torch.distributions.Normal(mean, torch.exp(logstd))
                    samples = dist.rsample((batch_size,))
                    samples  = self.inverse(self.split_zs(samples), max_iter=max_iter).cpu()
                    samples_list.append(samples)
                samples = torch.cat(samples_list, dim=0)
                return samples
            else:
                prior = self.prior()
                z = prior.rsample((batch_size,))
                zs = self.split_zs(z)
                return self.inverse(zs, max_iter=max_iter)

    def set_num_terms(self, n_terms):
        for block in self.stack:
            for layer in block.stack:
                layer.numSeriesTerms = n_terms



class LabelNet(torch.nn.Module):
    def __init__(self, nClasses=10, output_shape=[3, 32, 32]):
        super(LabelNet, self).__init__()
        self.net_head = torch.nn.Sequential(
            torch.nn.Linear(nClasses, int(torch.prod(torch.tensor(output_shape))) // 16),
            torch.nn.ELU()
        )
        self.transposed_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=output_shape[0], out_channels=output_shape[0], kernel_size=2, stride=2),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(in_channels=output_shape[0], out_channels=output_shape[0], kernel_size=2, stride=2),
            torch.nn.ELU()
        )
        self.output_shape = output_shape
    def forward(self, labels_onehot):
        head = self.net_head(labels_onehot).view(labels_onehot.size(0), self.output_shape[0], self.output_shape[1]//4,
                                                 self.output_shape[2]//4)
        out = self.transposed_conv(head)
        return out
#
# class conv_iResNet(nn.Module):
#     def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=2, inj_pad=0,
#                  coeff=.9, density_estimation=False, nClasses=None,
#                  numTraceSamples=1, numSeriesTerms=1,
#                  n_power_iter=5,
#                  block=conv_iresnet_block,
#                  actnorm=True, learn_prior=True,
#                  nonlin="relu"):
#         super(conv_iResNet, self).__init__()
#         assert len(nBlocks) == len(nStrides) == len(nChannels)
#         assert init_ds in (1, 2), "can only squeeze by 2"
#         self.init_ds = init_ds
#         self.ipad = inj_pad
#         self.nBlocks = nBlocks
#         self.density_estimation = density_estimation
#         self.nClasses = nClasses
#         # parameters for trace estimation
#         self.numTraceSamples = numTraceSamples if density_estimation else 0
#         self.numSeriesTerms = numSeriesTerms if density_estimation else 0
#         self.n_power_iter = n_power_iter
#
#         print('')
#         print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))
#         self.init_squeeze = Squeeze(self.init_ds)
#         self.inj_pad = injective_pad(inj_pad)
#         if self.init_ds == 2:
#            in_shape = downsample_shape(in_shape)
#         in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # adjust channels
#
#         self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides,
#                                                                         in_shape, coeff, block,
#                                                                         actnorm, n_power_iter, nonlin)
#
#         # make prior distribution
#         self._make_prior(learn_prior)
#         # make classifier
#         self._make_classifier(self.final_shape, nClasses)
#         assert (nClasses is not None or density_estimation), "Must be either classifier or density estimator"
#
#     def _make_prior(self, learn_prior):
#         dim = torch.prod(torch.tensor(self.in_shapes[0]))
#         self.prior_mu = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)
#         self.prior_logstd = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)
#
#     def _make_classifier(self, final_shape, nClasses):
#         if nClasses is None:
#             self.logits = None
#         else:
#             self.bn1 = nn.BatchNorm2d(final_shape[0], momentum=0.9)
#             self.logits = nn.Linear(final_shape[0], nClasses)
#
#     def classifier(self, z):
#         out = F.relu(self.bn1(z))
#         out = F.avg_pool2d(out, out.size(2))
#         out = out.view(out.size(0), out.size(1))
#         return self.logits(out)
#
#     def prior(self):
#         return distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))
#
#     def logpz(self, z):
#         return self.prior().log_prob(z.view(z.size(0), -1)).sum(dim=1)
#
#     def _make_stack(self, nChannels, nBlocks, nStrides, in_shape, coeff, block,
#                     actnorm, n_power_iter, nonlin):
#         """ Create stack of iresnet blocks """
#         block_list = nn.ModuleList()
#         in_shapes = []
#         for i, (int_dim, stride, blocks) in enumerate(zip(nChannels, nStrides, nBlocks)):
#             for j in range(blocks):
#                 in_shapes.append(in_shape)
#                 block_list.append(block(in_shape, int_dim,
#                                         numTraceSamples=self.numTraceSamples,
#                                         numSeriesTerms=self.numSeriesTerms,
#                                         stride=(stride if j == 0 else 1),  # use stride if first layer in block else 1
#                                         input_nonlin=(i + j > 0),  # add nonlinearity to input for all but fist layer
#                                         coeff=coeff,
#                                         actnorm=actnorm,
#                                         n_power_iter=n_power_iter,
#                                         nonlin=nonlin))
#                 if stride == 2 and j == 0:
#                     in_shape = downsample_shape(in_shape)
#
#         return block_list, in_shapes, in_shape
#
#     def get_in_shapes(self):
#         return self.in_shapes
#
#     # def inspect_singular_values(self):
#     #     i = 0
#     #     j = 0
#     #     params = [v for v in self.state_dict().keys()
#     #               if "bottleneck" in v and "weight_orig" in v
#     #               and not "weight_u" in v
#     #               and not "bn1" in v
#     #               and not "linear" in v]
#     #     print(len(params))
#     #     print(len(self.in_shapes))
#     #     svs = []
#         # for param in params:
#         #   input_shape = tuple(self.in_shapes[j])
#         #   # get unscaled parameters from state dict
#         #   convKernel_unscaled = self.state_dict()[param].cpu().numpy()
#         #   # get scaling by spectral norm
#         #   sigma = self.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
#         #   convKernel = convKernel_unscaled / sigma
#         #   # compute singular values
#         #   input_shape = input_shape[1:]
#         #   fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
#         #   t_fft_coeff = np.transpose(fft_coeff)
#         #   D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
#         #   Dflat = np.sort(D.flatten())[::-1]
#         #   print("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
#         #   svs.append(Dflat[0])
#         #   if i == 2:
#         #     i = 0
#         #     j+= 1
#         #   else:
#         #     i+=1
#         # return svs
#
#     def forward(self, x, ignore_logdet=False):
#         """ iresnet forward """
#         if self.init_ds == 2:
#             x = self.init_squeeze.forward(x)
#
#         if self.ipad != 0:
#             x = self.inj_pad.forward(x)
#
#         z = x
#         traces = []
#         for block in self.stack:
#             z, trace = block(z, ignore_logdet=ignore_logdet)
#             traces.append(trace)
#
#         # no classification head
#         if self.density_estimation:
#             # add logdets
#             tmp_trace = torch.zeros_like(traces[0])
#             for k in range(len(traces)):
#                 tmp_trace += traces[k]
#
#             logpz = self.logpz(z)
#             return z, logpz, tmp_trace
#
#         # classification head
#         else:
#             logits = self.classifier(z)
#             return logits, z
#
#     def inverse(self, z, max_iter=10):
#         """ iresnet inverse """
#         with torch.no_grad():
#             x = z
#             for i in range(len(self.stack)):
#                 x = self.stack[-1 - i].inverse(x, maxIter=max_iter)
#
#             if self.ipad != 0:
#                 x = self.inj_pad.inverse(x)
#
#             if self.init_ds == 2:
#                 x = self.init_squeeze.inverse(x)
#         return x
#
#     def sample(self, batch_size, max_iter=10):
#         """sample from prior and invert"""
#         with torch.no_grad():
#             # only send batch_size to prior, prior has final_shape as attribute
#             samples = self.prior().rsample((batch_size,))
#             samples = samples.view((batch_size,) + self.final_shape)
#             return self.inverse(samples, max_iter=max_iter)
#
#     def set_num_terms(self, n_terms):
#         for block in self.stack:
#             for layer in block.stack:
#                 layer.numSeriesTerms = n_terms


# if __name__ == "__main__":
#     scale = 1.
#     loc = 0.
#     base_distribution = distributions.Uniform(0., 1.)
#     transforms_1 = [distributions.SigmoidTransform().inv, distributions.AffineTransform(loc=loc, scale=scale)]
#     logistic_1 = distributions.TransformedDistribution(base_distribution, transforms_1)
#
#     transforms_2 = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
#     logistic_2 = distributions.TransformedDistribution(base_distribution, transforms_2)
#
#     x = torch.zeros(2)
#     print(logistic_1.log_prob(x), logistic_2.log_prob(x))
#     1/0
#
#     diff = lambda x, y: (x - y).abs().sum()
#     batch_size = 13
#     channels = 3
#     h, w = 32, 32
#     in_shape = (batch_size, channels, h, w)
#     x = torch.randn((batch_size, channels, h, w), requires_grad=True)
#
#     block = conv_iresnet_block(in_shape[1:], 32, stride=1, actnorm=True)
#     out, tr = block(x)#, ignore_logdet=True)
#     print("block")
#     for i in range(10):
#         x_re = block.inverse(out, i)
#         print(i, diff(x, x_re))
#
#     steps = 4
#     int_dim = 32
#     sb = scale_block(steps, in_shape[1:], int_dim, True, 5, 1, True, .9, False, True, True)
#
#     [z1, z2], tr = sb(x)#, ignore_logdet=True)
#     print("scale block")
#     for i in range(1):
#         x_re = sb.inverse(z1, z2, i)
#         print(i, diff(x, x_re))
#
#     resnet = conv_iResNet(in_shape[1:], [4, 4, 4], [1, 2, 2], [32, 32, 32],
#                           init_ds=2, density_estimation=True, actnorm=True)
#     print(resnet.final_shape)
#     z, lpz, tr = resnet(x)#, ignore_logdet=True)
#     for i in range(1):
#         x_re = resnet.inverse(z, i)
#         print("{} iters error {}".format(i, (x - x_re).abs().sum()))
#
#     resnet = multiscale_conv_iResNet(in_shape[1:], [4, 4, 4], [1, 2, 2], [32, 32, 32],
#                                      True, 0, .9, True, None, True, 1, 5, True)
#     out, logpz, tr = resnet(x)#, ignore_logdet=True)
#     print(logpz)
#     print(tr)
#     print([o.size() for o in out])
#     print(resnet.z_shapes())
#     sample = resnet.sample(33)
#     print(sample.size())
#     print('multiscale')
#     for i in range(20):
#         x_re = resnet.inverse(out, i)
#         print(i, diff(x, x_re))
#
