import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.utils.data

CUDA = torch.cuda.is_available()


class MagnusNet(nn.Module):
    
    @staticmethod
    def no_dim_reduction_conv(in_channels, out_channels, padding=1, bias=True):
        return nn.Conv2d(in_channels, out_channels, 3, 1, padding, bias=bias)
    
    @staticmethod
    def half_dim_reduction_conv(in_channels, out_channels, padding=1, bias=True):
        return nn.Conv2d(in_channels, out_channels, 4, 2, padding, bias=bias)
    
    def __init__(self, n_features):
        super(MagnusNet, self).__init__()
        self.n_features = n_features
        self.convnet = nn.Sequential(
            MagnusNet.half_dim_reduction_conv(3, n_features),
            nn.BatchNorm2d(n_features),
            nn.LeakyReLU(0.1, True),
            
            MagnusNet.half_dim_reduction_conv(n_features, n_features * 2, 0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(n_features*2),
            nn.LeakyReLU(0.1, True),
            
            MagnusNet.half_dim_reduction_conv(n_features * 2, n_features * 4),
            nn.BatchNorm2d(n_features*4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_features * 4, n_features, 5),
        )
        self.linearnet = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(n_features, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, self.n_features)
        x = self.linearnet(x)
        return x
    
    @staticmethod
    def net_instance(n_features):
        instance = MagnusNet(n_features)
        if CUDA:
            instance.cuda()
        return instance


parser = argparse.ArgumentParser(description='neurohack')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='ADAM momentum (default: 0.9)')
parser.add_argument('--n-features', type=int, default=64, metavar='N',
                    help='n_features (default: 64)')

parser.add_argument('--save', default='false')

args = parser.parse_args()

batch_size = args.batch_size
n_epochs = args.epochs
n_features = args.n_features

dataset = datasets.ImageFolder(root='neurodata/neuro-train',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ])
                                   )
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

testset = datasets.ImageFolder(root='neurodata/neuro-test',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ])
                                   )

testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

model = MagnusNet.net_instance(n_features)

learning_rate = args.lr
beta_one = args.momentum
beta_two = 0.999
optimizer = optim.Adam(model.parameters(), learning_rate, (beta_one, beta_two))

criterion = nn.MSELoss()


def train(epoch):
    model.train()
    for i, (x, label) in enumerate(dataloader):
        label = label.float()
        if CUDA:
            x, label = x.cuda(), label.cuda()
        x, label = Variable(x), Variable(label)
        model.zero_grad()
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(x), len(dataloader.dataset),
                   100. * i / len(dataloader), loss.data[0]))
            
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for i, (x, label) in enumerate(testloader):
        int_label = torch.LongTensor(label)
        label = label.float()
        if CUDA:
            x, label = x.cuda(), label.cuda()
        x, label = Variable(x, volatile=True), Variable(label)
        output = model(x)
        test_loss += criterion(output, label)
        output_tensor = output.data
        print(output_tensor)
        output_tensor = output_tensor.cpu().apply_(lambda x: 0.0 if x < 0.5 else 1.0)
        onp = output_tensor.numpy().flatten()
        lnp = int_label.cpu().numpy()
        print(onp, ' ', lnp)
        correct += (abs(onp - lnp) < 0.02).sum()
        total += int_label.size()[0]
    test_loss = test_loss.data.cpu().numpy()[0] / total
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}\n'.format(
        test_loss, correct, total))
    
    
if __name__ == '__main__':
    import time
    start = time.time()
    for epoch in range(1, n_epochs):
        train(epoch)
        test()
    from PIL import Image
    import numpy as np
    
    model.eval()
    
    x = Image.open('neurodata/neuro-train/1/1.png')
    x.load()
    data = np.asarray(x, 'int32')
    data = torch.from_numpy(data).float()
    data = Variable(data).view(1, 3, 84, 84)
    
    n_x = Image.open('neurodata/neuro-train/0/adrey2.png')
    n_x.load()
    n_data = np.asarray(n_x, 'int32')
    n_data = torch.from_numpy(n_data).float()
    n_data = Variable(n_data).view(1, 3, 84, 84)
    
    if CUDA:
        data = data.cuda()
        n_data = n_data.cuda()
    print(model(data))
    print(model(n_data))
    finish = time.time()
    print('Execution time ', finish-start, ' s')
    print(args.save)
    if args.save == 'true':
        torch.save(model.state_dict(), 'trained.pt')
