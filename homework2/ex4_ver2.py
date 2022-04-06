import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
import tqdm
from PIL import Image

# Example https://www.kaggle.com/jaeboklee/pytorch-cat-vs-dog

class SlugDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            if 'lusitania' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]

def train():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    train_dir = 'images/original'
    train_files = os.listdir(train_dir)

    lusitania_files = [tf for tf in train_files if 'lusitania' in tf]
    limax_files = [tf for tf in train_files if 'limax' in tf]

    lusitanias = SlugDataset(lusitania_files, train_dir, transform = data_transform)
    limaxs = SlugDataset(limax_files, train_dir, transform = data_transform)

    slugs = ConcatDataset([lusitanias, limaxs])
    dataloader = DataLoader(slugs, batch_size = 32, shuffle=True, num_workers=4)

    samples, labels = iter(dataloader).next()

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    # Download model 30.8M/30.8M [00:02<00:00, 11.4MB/s]
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

    epochs = 30
    itr = 1
    p_itr = 5
    model.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for samples, labels in dataloader:

            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            print('[Iteration {}] Train Loss: {:.4f}'.format(itr, total_loss))
            if itr%p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0
                
            itr += 1
   
    plt.plot(loss_list, label='Cross-Entropy loss')
    plt.plot(acc_list, label='Accuracy')
    plt.legend()
    plt.title('Training loss and accuracy')
    plt.savefig('ex4_loss_accuracy.png')

    filename_pth = 'ex4_model.pth'
    torch.save(model, filename_pth)

def test():
    test_dir = 'images/test'
    test_files = os.listdir(test_dir)
    device = 'cuda'
    model_file = 'ex4_model.pth'
    model = torch.load(model_file)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    testset = SlugDataset(test_files, test_dir, mode='test', transform = test_transform)
    testloader = DataLoader(testset, batch_size = 32, shuffle=False, num_workers=4)
    
    fn_list = []
    pred_list = []
    for x, fn in testloader:
        with torch.no_grad():
            x = x.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            fn_list += [n[:-4] for n in fn]
            pred_list += [p.item() for p in pred]

    submission = pd.DataFrame({"id":fn_list, "label":pred_list})
    submission.to_csv('ex4_result.csv', index=False)

    samples, _ = iter(testloader).next()
    samples = samples.to(device)
    fig = plt.figure(figsize=(24, 16))
    fig.tight_layout()
    output = model(samples[:24])
    pred = torch.argmax(output, dim=1)
    pred = [p.item() for p in pred]
    ad = {0:'limax', 1:'lusitania'}
    for num, sample in enumerate(samples[:24]):
        plt.subplot(4,5,num+1)
        plt.title(ad[pred[num]])
        plt.axis('off')
        sample = sample.cpu().numpy()
        plt.imshow(np.transpose(sample, (1,2,0)))
        
    plt.savefig('ex4_prediction.png')
    plt.show()

if __name__ == '__main__':
    train()
    test()
