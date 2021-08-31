'''Train dataset with PyTorch. code from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from mobilenet_classification_model import *
from resnet import resnet50, resnet101
from classification_data_loader import *
from utils import progress_bar


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
#create a pipeline for the dataset
train_pipe = TrainPipeline(256, 4, 0)

#create an iterator after building the pipeline
train_pipe.build()
train_loader = DALIGenericIterator(train_pipe, ['data', 'label'], 200000)

test_pipe = TestPipeline(256, 4, 0)
test_pipe.build()
test_loader = DALIGenericIterator(test_pipe, ['data', 'label'], 30000)

# Model
print('==> Building model..')
net = resnet50()
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, epochs=100, steps_per_epoch=200000//256)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    precision = 0
    recall = 0
    false_negatives = 0.0
    true_positives = 0.0
    false_positives = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs = data[0]['data']
        targets = one_hot(data[0]['label'])
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        loss = margin_loss(epoch, torch.sigmoid(outputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        temp_true_positives, temp_false_positives, temp_false_negatives = get_val(outputs, targets)
        total += targets.size(0)
        false_negatives += temp_false_negatives
        true_positives += temp_true_positives
        false_positives += temp_false_positives
        if true_positives + false_positives != 0:
           precision = (true_positives/(true_positives + false_positives))
        else:
             precision = 0.0
        if true_positives + false_negatives != 0:
           recall = (true_positives/(true_positives + false_negatives))
        else:
             recall = 0.0
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | precision: %.3f%% recall: %.3f%% %.3f%%'
                     % (train_loss/(batch_idx+1), 100.*precision, 100.*recall, torch.mean(torch.sigmoid(outputs))))
    train_loader.reset()    


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    precision = 0
    recall = 0
    false_negatives = 0.0
    true_positives = 0.0
    false_positives = 0.0
    global best_acc
    with torch.no_grad():
         for batch_idx, data in enumerate(test_loader):
             inputs = data[0]['data']
             targets = one_hot(data[0]['label'])
             inputs, targets = inputs.to(device), targets.to(device)
             outputs = net(inputs)
             #loss = criterion(outputs, targets)
             loss = margin_loss(epoch, torch.sigmoid(outputs), targets)
             test_loss += loss.item()
             temp_true_positives, temp_false_positives, temp_false_negatives = get_val(outputs, targets)
             total += targets.size(0)
             false_negatives += temp_false_negatives
             true_positives += temp_true_positives
             false_positives += temp_false_positives
             if true_positives + false_positives != 0:
                precision = (true_positives/(true_positives + false_positives))
             else:
                  precision = 0.0
             if true_positives + false_negatives != 0:
                recall = (true_positives/(true_positives + false_negatives))
             else:
                  recall = 0.0
             progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | precision: %.3f%% recall: %.3f%% %.3f%%'
                               % (test_loss/(batch_idx+1), 100.*precision, 100.*recall, torch.mean(torch.sigmoid(outputs))))
         print(precision)
         print(recall)
         if (precision*recall/(precision+recall) >= best_acc):
            print('Saving..')
            print(precision*recall/(precision+recall), best_acc)
            state = {
                     'model': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_Scheduler': scheduler.state_dict(),
                     'epoch': epoch,
                     'precision': precision,
                     'recall': recall,
                    }
            if not os.path.isdir('checkpoint'):
               os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = precision*recall/(precision+recall) 
         test_loader.reset()

def get_val(outputs, targets):
    outputs = (torch.sigmoid(outputs) > 0.5).float()
    true_positives = (outputs*targets).sum().item()
    false_positives = (outputs*(1-targets)).sum().item()
    false_negatives = ((1-outputs)*targets).sum().item()
    return true_positives, false_positives, false_negatives

def margin_loss(epoch, class_activations, target, lambda_=0.3,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1)
    right = F.relu(class_activations - negative_margin).view(batch_size, -1)
    margin_loss = 5*target*left + lambda_ *(1-target)*right
    if epoch == 0:
       margin_loss = margin_loss.sum(dim=1).mean() + 0.3*torch.mean(class_activations)
    if epoch !=0:
       margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss
    

def one_hot(targets, num_classes=528):
    label = torch.zeros(targets.size(0), num_classes+1).scatter_(1, targets.long(), 1.)
    label = label[:, 1:]
    return label

'''for epoch in range(start_epoch, 100):
    train(epoch)
    test(epoch)'''
net.load_state_dict(torch.load('checkpoint/ckpt.pth')['model'])
test(0)
