"""
Build and save a model for the cat and dog image classification
"""

import argparse
import math
import time

# from mnist import MNIST
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor, Resize, Lambda
from oxford_iiit_pet import OxfordIIITPet

import collections
collections.Iterable = collections.abc.Iterable

try:
    import torch_nnpa
except ImportError:
    print("Error importing torch_nnpa")

MODEL_FILENAME = 'model_cnn.pt'


class ImageDataset(Dataset):
    def __init__(self, images, labels, normalize=False):
        self.images = images
        self.labels = labels
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.uint8)
        image = torch.reshape(image, [1, 28, 28])
        if self.normalize is not None:
            normalized = image.float() / 255.0
            image = (normalized - 0.1307) / 0.3081

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class Net(nn.Module):
    def __init__(self, resize=128):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 1)
        self.conv5 = nn.Conv2d(64, 128, 4, 1)
        self.conv6 = nn.Conv2d(128, 128, 4, 1)
        self.conv7 = nn.Conv2d(128, 256, 3, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        if resize==128:
            linear_nodes = 2304  # 256 * 3 * 3
        if resize==256:
            linear_nodes = 30976  # 256 * 11 * 11
        self.fc1 = nn.Linear(linear_nodes, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # input: 128x128
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # output: 64x64

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # output: 32x32

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # output: 16x16

        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # output: 8x8

        # print(f"size: {x.size()}  prod: {math.prod(x.size()[1:])}")

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model: nn.Module, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        test_infer_loop(model, device, test_loader)


def infer(model, device, test_loader):
    model.eval()
    start_time = time.time()
    with torch.inference_mode():
        test_infer_loop(model, device, test_loader)
    end_time = time.time()
    print('Infer time: {:.4f}'.format(end_time - start_time))


def test_infer_loop(model, device, test_loader):
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # print(f"Output: {output}")
        # print(f"Output: {output}")
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def infer_once(model, device, input_data, target):
    with torch.inference_mode():
        input_data, target = input_data.to(device), target.to(device)
        output = model(input_data)
        test_loss = F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
    return correct, test_loss


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='Number of DataLoader workers (default 1)')
    parser.add_argument('--resize', type=int, default=128, metavar='NxN',
                        help='Resize the images to this size (default 128)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save the trained model. Only applicable when training.')
    parser.add_argument('--model-path', default='model_cnn.pt', help='path to save or load model file')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Only test the saved model - do not train')
    args = parser.parse_args()
    return args


def test_setup(batch_size=4, resize=128, num_workers=4, device='cpu', model_path=MODEL_FILENAME):
    """ For running inference from another Python module """
    print(f"Setting up testing on device {device}")
    try:
        device = torch.device(device)
    except RuntimeError:
        print(f"Error using device {device} - proceeding with cpu instead")
        device = torch.device('cpu')

    oxford_pet_test = OxfordIIITPet(
        root='data/',
        split="test",
        target_types="binary-category",
        download=True,
        transform=torchvision.transforms.Compose([
            Resize((resize, resize)),
            ToTensor(),
        ]),
    )
    test_loader = DataLoader(oxford_pet_test,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    model = Net(resize=resize).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"{device} setup complete")
    return model, device, test_loader

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    device = torch.device('cpu')

    # mnist_data = MNIST('./data')
    # train_images, train_labels = mnist_data.load_training()
    # test_images, test_labels = mnist_data.load_testing()
    print("Opening dataset...")
    oxford_pet_train = OxfordIIITPet(
        root='data/',
        split="trainval",
        target_types="binary-category",
        download=True,
        transform=torchvision.transforms.Compose([
            Resize((args.resize, args.resize)),
            ToTensor(),
        ]),
    )
    # print(oxford_pet_train[-1])

    oxford_pet_test = OxfordIIITPet(
        root='data/',
        split="test",
        target_types="binary-category",
        download=True,
        transform=torchvision.transforms.Compose([
            Resize((args.resize, args.resize)),
            ToTensor(),
        ]),
    )

    # train_dataset = ImageDataset(train_images, train_labels, normalize=True)
    # test_dataset = ImageDataset(test_images, test_labels, normalize=True)

    # train_kwargs = {'batch_size': args.batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}

    # train_loader = DataLoader(train_dataset, **train_kwargs)
    # test_loader = DataLoader(test_dataset, **test_kwargs)
    print("Loading dataset...")
    train_loader = DataLoader(oxford_pet_train,
                              batch_size=4,
                              shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(oxford_pet_test,
                              batch_size=4,
                              shuffle=True,
                              num_workers=args.num_workers)

    print("Initializing neural net...")
    model = Net(resize=args.resize).to(device)
    # print(f"Model: {Net()}")
    if args.infer:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        print("Starting inference...")
        infer(model, device, test_loader)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        print("Starting training...")
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), args.model_path)


if __name__ == '__main__':
    main()
