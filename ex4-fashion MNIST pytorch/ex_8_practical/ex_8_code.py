import argparse
import numpy as np
import torch as torch
import matplotlib.pyplot as plot
import torchvision
# import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler



class FirstNet(torch.nn.Module):
    def __init__(self,image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class SecondNet(torch.nn.Module):
    def __init__(self, image_size):
        super(SecondNet, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.bn2 = torch.nn.BatchNorm1d(50)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.image_size)
        x = self.bn1(F.relu(self.fc0(x)))
        x = self.bn2(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class ThirddNet(torch.nn.Module):
    def __init__(self, image_size):
        super(ThirddNet, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss=0;
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        avg_loss+=loss
        loss.backward()
        optimizer.step()
    return int(avg_loss/len(train_loader.dataset))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    ## Define our MNISTfashion Datasets (Images and Labels) for training and testing

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    ## Define our MNIST Datasets (Images and Labels) for training and testing
    train_dataset = datasets.FashionMNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.FashionMNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor())
    indices = list(range(len(train_dataset)))  # start with all the indices in training set
    split = 10000  # define the split size

    # Define your batch_size
    batch_size = 64

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your test set
    # operations shouldn't be computationally intensive or require batching.  We
    # also turn off shuffling, although that shouldn't affect your test set operations
    # either
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)



    # model = Net().to(device)
    model1 = FirstNet(image_size=28 * 28)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    model2 = SecondNet(image_size=28 * 28)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum)
    model3 = SecondNet(image_size=28 * 28)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=args.lr, momentum=args.momentum)

    # Y_train = np.zeros((100, D))
    X_new = np.zeros((10, 1))
    Y_train1=np.zeros((10,1))
    Y_test1=np.zeros((10,1))
    Y_train2 = np.zeros((10, 1))
    Y_test2 = np.zeros((10, 1))
    Y_train3 = np.zeros((10, 1))
    Y_test3 = np.zeros((10, 1))
    for epoch in range(1, args.epochs + 1):
        i=epoch-1
        X_new[i]=epoch
        # print('model 1')
        Y_train1[i]=train(args, model1, device, train_loader, optimizer1, epoch)
        Y_test1[i] =lossAvg(args, model1, device, validation_loader)
        # print('model2')
        Y_train2[i]=train(args, model2, device, train_loader, optimizer2, epoch)
        Y_test2[i]=lossAvg(args, model2, device, validation_loader)
        # print('model3')
        Y_train3[i]=train(args, model3, device, train_loader, optimizer3, epoch)
        Y_test3[i]=lossAvg(args, model3, device, validation_loader)

    # plot.xlim(0, 10)
    # plot.ylim(0, 1)
    plot.plot(X_new, Y_train1, "*")
    plot.plot(X_new, Y_test1, "*")
    plot.show()

    # plot.xlim(0, 10)
    # plot.ylim(0, 1)
    plot.plot(X_new, Y_train2, "*")
    plot.plot(X_new, Y_test2, "*")
    plot.show()

    # plot.xlim(0, 10)
    # plot.ylim(0, 1)
    plot.plot(X_new, Y_train3, "*")
    plot.plot(X_new, Y_test3, "*")
    plot.show()


    print('model 1')
    l1=test(args, model1, device, test_loader)
    print('model2')
    l2=test(args, model2, device, test_loader)
    print('model3')
    l3=test(args, model3, device, test_loader)

    # test_x = np.loadtxt("test_x")

    mypred=""
    if(l1>l2)&(l1>l3):
        mymodel=1
        mypred =pred(args,model1,device,test_loader)
    if(l2>l1)&(l2>l3):
        mymodel=2
        mypred = pred(args, model2, device, test_loader)
    if(l3>l1)&(l3>l2):
        mymodel=3
        mypred = pred(args, model3, device, test_loader)

    file = open('test.pred', 'w')
    file.write(mypred)
    file.close()





def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    return 100. * correct / len(test_loader.dataset)


def lossAvg(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def pred(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    d=""
    with torch.no_grad():
        for data, target in test_loader:
            d+="\n"
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(pred.item())
            # print("\n")
            item=pred.item()
            d+=str(item)
    # model.eval()
    # test_loss = 0
    # correct = 0
    # pred=""
    # with torch.no_grad():
    #     for data in test_loader:
    #         pred += "\n"
    #         data= data.to(device)
    #         output = model(data)
    #         # test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
    #         pred+= output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    return d



if __name__ == '__main__':
    main()

