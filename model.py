from __future__ import print_function
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from skimage import io, transform

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 80, 3, 1)
        self.conv2 = nn.Conv2d(80, 160, 3, 1)
        self.conv3 = nn.Conv2d(160, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1)

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)


        self.fc1 = nn.Linear(5408, 32)

        self.fcq1 = nn.Linear(27, 320)
        self.fcq2 = nn.Linear(320, 32)

        self.fc2 = nn.Linear(32, 13)
        self.fc3 = nn.Linear(13, 13)


    def forward(self, im_input, q_input, big_model):
        x1 = self.conv1(im_input)
        x1 = self.maxpool1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.maxpool2(x1)
        if big_model:
            x1 = self.conv4(x1)
            x1 = self.maxpool3(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc1(x1)
        x1 = F.tanh(x1)

        x2 = self.fcq1(q_input)
        x2 = F.tanh(x2)
        x2 = self.fcq2(x2)
        x2 = F.tanh(x2)
        combined_feature = torch.mul(x1, x2)  # [batch_size, embed_size]

        pred = self.fc2(combined_feature)
        pred = F.tanh(pred)
        pred = self.fc3(pred)
        return pred

class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.root_path = path
        self.image_path = os.path.join(self.root_path, 'images')
        self.question_path = os.path.join(self.root_path, 'questions.json')
        self.texts, self.answers, self.image_ids = self.read_questions(self.question_path)
        self.img_paths = self.extract_paths(self.image_path)
        self.transform = transform
        self.all_answers = self.read_answers(os.path.join(self.root_path, '../answers.txt'))
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.texts)
        self.text_seqs = tokenizer.texts_to_matrix(self.texts)
        self.answer_indices = [self.all_answers.index(a) for a in self.answers]

    def read_questions(self, path):
        with open(path, 'r') as file:
            qs = json.load(file)
        texts = [q[0] for q in qs]
        answers = [q[1] for q in qs]
        image_ids = [q[2] for q in qs]
        return texts, answers, image_ids
    
    def read_answers(self, path):
        with open(path, 'r') as file:
            all_answers = [a.strip() for a in file]
        return all_answers
    
    def extract_paths(self, dirctory):
        paths = {}
        for filename in os.listdir(dirctory):
            if filename.endswith('.png'):
                image_id = int(filename[:-4])
                paths[image_id] = os.path.join(dirctory, filename)
        return paths


    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = img_to_array(load_img(self.img_paths[self.image_ids[idx]]))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'question': torch.Tensor(self.text_seqs[idx])}
        target = torch.LongTensor([self.answer_indices[idx]])
        return sample, target

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (sample, label) in enumerate(train_loader):
        correct_train = 0
        img, ques, target = sample['image'].to(device), sample['question'].to(device), label.to(device)
        target = target.view(ques.shape[0])
        optimizer.zero_grad()
        output = model(img, ques, False)
        NLL = nn.CrossEntropyLoss()  
        loss = NLL(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_train += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(ques), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print("Accuracy: {}%\n".format(correct_train / len(ques)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (sample, label) in enumerate(test_loader):
            img, ques, target = sample['image'].to(device), sample['question'].to(device), label.to(device)
            target = target.view(ques.shape[0])
            output = model(img, ques, False)
            NLL = nn.CrossEntropyLoss()
            test_loss += NLL(output, target).item()# sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Easy-VQA Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu', default=False,
                        help='GPU usage (default: False))')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.90441555, 0.90574956, 0.89646965), (0.19336925, 0.18681642, 0.20578428))
        ])

    dataset1 = MyDataset('./data/train', transform)
    dataset2 = MyDataset('./data/test', transform)
    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=False, num_workers=4, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, shuffle=False, num_workers=4, **test_kwargs)
    print("The length of train_loader is {}, and test_loader is {}".format(len(train_loader), len(test_loader)))

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
