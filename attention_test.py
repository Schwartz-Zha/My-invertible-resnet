import torch
import torchvision
from torchvision import transforms
from models.model_utils import squeeze
from models.inv_attention import InvAttention_dot2
import argparse
import time
from torch.autograd import Variable
import numpy as np
import os
from spectral_norm_fc import spectral_norm_fc

class Attention_Test(torch.nn.Module):
    def __init__(self):
        super(Attention_Test, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.attention_layer = InvAttention_dot2(12)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = self.attention_layer.forward(x)[0]
        return x
    def inverse(self, x):
        x = self.attention_layer.inverse(x)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip

class Conv_Test(torch.nn.Module):
    def __init__(self):
        super(Conv_Test, self).__init__()
        self.squeeze_layer = squeeze(2)
        self.conv_layer = spectral_norm_fc(torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
    def forward(self, x):
        x = self.squeeze_layer.forward(x)
        x = x + self.conv_layer.forward(x)
        return x
    def inverse(self, y, maxIter=100):
        x = y
        for i in range(maxIter):
            x = y - self.conv_layer.forward(x)
        x = self.squeeze_layer.inverse(x)
        return x
    def inspect_lip(self, x, eps=0.00001):
        x = self.squeeze_layer(x)
        dx = x * eps
        y1 = self.attention_layer.res_branch.forward(x)
        y2 = self.attention_layer.res_branch.forward(x + dx)
        lip = torch.dist(y2, y1) / torch.dist((x + dx), x)
        return lip

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='results/invattention_test')
parser.add_argument('--show_image', type=bool, default=True)
parser.add_argument('--model', type=str, default='attention')

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def main():
    args = parser.parse_args()
    try_make_dir(args.save_dir)
    use_cuda = torch.cuda.is_available()
    dens_est_chain = [
        lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        lambda x: x / 256.,
        lambda x: x - 0.5
    ]
    test_chain = [transforms.ToTensor()]
    train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                   transforms.RandomCrop(32),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor()]
    transform_train = transforms.Compose(train_chain + dens_est_chain)
    transform_test = transforms.Compose(test_chain + dens_est_chain)


    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_subset = torch.utils.data.Subset(trainset, list(range(1000)))
    test_subset = torch.utils.data.Subset(testset, list(range(1000)))

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64,
                                              shuffle=True, num_workers=2,drop_last=True,
                                              worker_init_fn=np.random.seed(1234))
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=64,
                                             shuffle=False, num_workers=2,drop_last=True,
                                             worker_init_fn=np.random.seed(1234))
    if args.model == 'attention':
        model = Attention_Test()
    else:
        model = Conv_Test()
    if use_cuda:
        model.cuda()

    target = torch.randn([64, 12, 16, 16])
    target = Variable(target)
    if use_cuda:
        target = target.cuda()


    criterion = torch.nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)

    elapsed_time = 0.
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            optim.zero_grad()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs, requires_grad=True)
            output = model.forward(inputs)
            loss = criterion(output, target)
            loss.backward()
            optim.step()

        epoch_time = time.time() - start_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Test if the model is invertible
    batch = None

    for batch_idx, (inputs, targets) in enumerate(testloader):
        batch = inputs
        batch = Variable(batch, requires_grad=True)
        if use_cuda:
            batch = batch.cuda()
        break

    model = model.eval()
    img_dir = os.path.join(args.save_dir, 'ims')
    try_make_dir(img_dir)

    output = model(batch)
    inverse_input = model.inverse(output)
    if args.show_image:
        torchvision.utils.save_image(batch.cpu(),
                                     os.path.join(img_dir, "data.jpg"),
                                     8, normalize=True)
        torchvision.utils.save_image(inverse_input.cpu(),
                                     os.path.join(img_dir, "recon.jpg"),
                                     8, normalize=True)

    print("lipschitz constant of atttention: " + str(model.inspect_lip(batch)))



if __name__ == '__main__':
    main()


