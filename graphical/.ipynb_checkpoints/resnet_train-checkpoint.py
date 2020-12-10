import pandas as pd
import numpy as np 

import torch 
import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
from torch import optim

from PIL import Image
from tqdm import tqdm

import os

import matplotlib.pyplot as plt

from GPUtil import showUtilization as gpu_usage

from datetime import datetime


# optimize if fixed size input
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# use gpu if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print()
print("Device: ", device)



# *********************************** load the data ***********************************

print()
print('Init dataloader...')


# where the data is located, must contains two folders 'train' and 'valid', each containing subfolders w.r.t to categories
path = '/dlabdata1/lugeon/websites_alexa_20_000_screenshots/'
#path = '/dlabdata1/lugeon/websites_alexa_mp2_1500_9cat_screenshots/'

batch_size = 200

# dimensions of the screeshots
valid_xdim = 640 # 640
valid_ydim = 360 # 360
crop_factor = 0.7

crop_dim = [int(crop_factor * valid_ydim), int(crop_factor * valid_xdim)]

color_jitter = transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0.2, hue=0.2)

random_crop = transforms.RandomCrop(size=crop_dim) # crop transform for data augmentation
five_crop = transforms.FiveCrop(size=crop_dim) 

tensorize = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

stack_norm_tensorize = transforms.Lambda(lambda crops: torch.stack([normalize(tensorize(crop)) for crop in crops]))


data_transforms = {
    'train': transforms.Compose([random_crop, color_jitter, tensorize, normalize]),
    'valid': transforms.Compose([five_crop, stack_norm_tensorize])
}




images_dict = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) for x in ['train', 'valid']}

dataloaders_dict = {x: torch.utils.data.DataLoader(images_dict[x], 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   num_workers=4,
                                                   pin_memory=True) for x in ['train', 'valid']}
                                                   


# *********************************** build model ***********************************
    

out_dim = 16 # number of classes
features_dim = 512 # number of features before the classifier

class Webnet(nn.Module):
    def __init__(self):
        super(Webnet, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = torch.nn.Linear(features_dim, features_dim)
        self.fc2 = torch.nn.Linear(features_dim, out_dim)
        
        #self.drop = torch.nn.Dropout(0.7)

    def forward(self, x):
        x = self.features(x).reshape(-1, features_dim)
        #x = self.drop(x)
        x = self.fc1(x)
        #x = self.drop(x)
        x = self.fc2(F.relu(x))
        return x
    

    
    
    
    
# *********************************** print classes and parameters ***********************************    

# computing number of samples 
counts_train = [len(os.listdir(path + 'train/' + cat)) for cat in os.listdir(path + 'train/')]
counts_valid = [len(os.listdir(path + 'valid/' + cat)) for cat in os.listdir(path + 'valid/')]

cats = [cat for cat in os.listdir(path + 'train/')]

sum_train = sum(counts_train)
sum_valid = sum(counts_valid)

# weights for unbalanced dataset (only computed on train)
weights = torch.Tensor([1 / (x / sum_train) for x in counts_train]).to(device)
weights = weights / weights.sum() * out_dim

print()
print('*** Train samples ***')
for cat, n in zip(cats, counts_train):
    print('  {:<11} {:>7}'.format(cat, n))
print('  {:<11} {:>7}'.format('Total', sum_train))

print()
print('*** Valid samples ***')
for cat, n in zip(cats, counts_valid):
    print('  {:<11} {:>7}'.format(cat, n))
print('  {:<11} {:>7}'.format('Total', sum_valid))

print()
print('*** Weights ***')
for cat, weight in zip(cats, weights):
    print('  {:<11} {:7.3f}'.format(cat, weight))
print('  {:<11} {:7.3f}'.format('Total', sum(weights)))

print()
print('Number of classes: {}'.format(len(cats)))






# *********************************** training ***********************************

nb_epochs = 100

model = Webnet().to(device)

nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: {}".format(nb_trainable_params))

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), 1e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(nb_epochs / 2), gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

train_acc_hist = []
train_loss_hist = []
test_acc_hist = []
test_loss_hist = []

try:

    for e in range(nb_epochs):
        
        print()
        print('Epoch {}'.format(e))
        
        # tune if we want to train the features extraction
        for p in model.features.parameters():
            p.requires_grad = True
            
        # first we train, then we test
        for phase in ['train', 'valid']:
            
            # split again batch size if there is data augmentation
            augmentation_batch = batch_size
                                
            if phase == 'train':
                model.train() 
            else:
                model.eval()
                augmentation_batch = int(augmentation_batch / 5) # five crop for validation
                
            # accumulate loss and acc through the batches
            running_loss = 0.0
            running_corrects = 0
            

            for data in tqdm(iter(dataloaders_dict[phase])):

                inputs_ = data[0]
                targets_ = data[1]
                
                for inputs, targets in zip(inputs_.split(augmentation_batch), targets_.split(augmentation_batch)):
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # reset all gradients to zero
                    optimizer.zero_grad()

                    # only activate autograd if train phase
                    with torch.set_grad_enabled(phase == 'train'):

                        # if valid phase, then flatten the five crops
                        if phase == 'valid':
                            bs, ncrops, c, h, w = inputs.size()
                            outputs_ = model(inputs.view(-1, c, h, w)) # output for each crop
                            outputs = outputs_.view(bs, ncrops, -1).mean(1) # mean over the crops

                        else:
                            outputs = model(inputs)

                        loss = criterion(outputs, targets)

                        # updates parameters if train phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                        # class prediction is the one that maximize output
                        values, preds = torch.max(outputs, 1)

                        # update running loss and accuracy
                        running_loss += loss.detach() * inputs.size(0) # divide by total number of samples at the end
                        running_corrects += torch.sum(preds == targets.data).detach() 
                
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders_dict[phase].dataset)
            
            if phase == 'train':
                train_loss_hist += [epoch_loss]
                train_acc_hist += [epoch_acc]
            else:
                test_loss_hist += [epoch_loss]
                test_acc_hist += [epoch_acc]
                        
            print('--- {} loss: {:.3f}    accuracy: {:.3f}'.format(phase, epoch_loss, epoch_acc))
                
        scheduler.step(epoch_acc) # testing accuracy
        
finally:
    
    now = datetime.now()
    now_string = now.strftime("%d-%m-%y_%H-%M")
    
    torch.save(model, '/scratch/lugeon/' + "webnet_" + now_string)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))


    ax1.set_xlabel('Epochs', size=15)
    ax1.set_ylabel('Loss', size=15)
    ax1.plot(train_loss_hist, color='lightskyblue', label='train loss')
    ax1.plot(test_loss_hist, color='steelblue', label='test loss')
    #ax1.set_ylim([0,3])
    ax1.legend(loc='upper left')


    ax2 = ax1.twinx()

    ax2.set_ylabel('Accuracy', size=15)
    ax2.plot(train_acc_hist, color='sandybrown', label='train acc')
    ax2.plot(test_acc_hist, color='sienna', label='test acc')
    ax2.set_ylim([0,1])


    ax2.legend(loc='upper right')
    plt.savefig('train_plots/' + 'training_' + now_string)
    