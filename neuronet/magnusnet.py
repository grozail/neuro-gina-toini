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
    def no_dim_reduction_conv(in_channels, out_channels, padding=1, bias=False):
        return nn.Conv2d(in_channels, out_channels, 3, 1, padding, bias=bias)
    
    @staticmethod
    def half_dim_reduction_conv(in_channels, out_channels, padding=1, bias=False):
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
            nn.Conv2d(n_features * 4, n_features * 4, 5),
        )
        self.linearnet = nn.Sequential(
            nn.Linear(n_features * 4, n_features),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(n_features, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, self.n_features*4)
        x = self.linearnet(x)
        return x
    
    @staticmethod
    def net_instance(n_features):
        instance = MagnusNet(n_features)
        if CUDA:
            instance.cuda()
        return instance

batch_size = 3
n_epochs = 10
n_features = 64

dataset = datasets.ImageFolder(root='/workspace/ilya/neuro-gina-toini/neurodata/neuro-train',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ])
                                   )
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

testset = datasets.ImageFolder(root='/workspace/ilya/neuro-gina-toini/neurodata/neuro-test',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ])
                                   )

testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

model = MagnusNet.net_instance(n_features)

learning_rate = 0.01
beta_one = 0.81
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
    for i, (x, label) in enumerate(testloader):
        if CUDA:
            x, label = x.cuda(), label.cuda()
        x, label = Variable(x, volatile=True), Variable(label)
        output = model(x)
        test_loss += criterion(output, label)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    
    
if __name__ == '__main__':
    for epoch in range(1, n_epochs):
        train(epoch)
        test()
