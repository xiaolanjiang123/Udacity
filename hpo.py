#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
  
        test_loss += loss.item() * data.size(0)
        temp, pred = torch.max(output,1)
        correct += torch.sum(pred == target.data)

    total_loss = test_loss//len(test_loader.dataset)
    total_acc = correct.double()//len(test_loader.dataset)

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    loss_count = 0
    least_loss = 99999999999
    epoch =50
    
    for e in range(epoch):
        model.train()

        train_loss = 0.0
        correct = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp, pred = torch.max(output,1)
            train_loss += loss.item() * data.size(0)
            correct+= torch.sum(pred == target.data)
        epoch_loss = train_loss // len(train_loader)
        epoch_accuracy = correct.double() // len(train_loader)
        logger.info('epoch{}, train loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(e,
                                                                                epoch_loss,
                                                                            epoch_accuracy,
                                                                                least_loss))
                    
        model.eval()

        train_loss = 0.0
        correct = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp, pred = torch.max(output,1)
            train_loss += loss.item() * data.size(0)
            correct+= torch.sum(pred == target.data)
        epoch_loss = train_loss // len(validation_loader)
        epoch_accuracy = correct // len(validation_loader)
        if epoch_loss<least_loss:
            least_loss = epoch_loss
        else:
            loss_count+=1
            
        logger.info('epoch{}, validation loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(e,
                                                                                epoch_loss,
                                                                                epoch_accuracy,
                                                                                least_loss))
        if loss_count == 1:
            break

    return model
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    
def net():
    model = models.resnet18(pretrained = True)
    for p in model.parameters():
        p.requires_grad = False
        nfeatures = model.fc.in_features
    model.fc=nn.Sequential(nn.Linear(nfeatures,256),
                           nn.ReLU(inplace = True),
                           nn.Linear(256,256),
                           nn.ReLU(inplace = True),
                           nn.Linear(256,256),
                           nn.ReLU(inplace = True),
                           nn.Linear(256,133))
    return model
                                     
        
    

    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''


def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    img_size = 224
    train_trans = transforms.Compose([transforms.Resize(img_size),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor()]) 
    test_trans = transforms.Compose([transforms.Resize(img_size),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor()]) 
    data_train = torchvision.datasets.ImageFolder(root = train_path, transform = train_trans)
    train_data_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size)
    data_test = torchvision.datasets.ImageFolder(root = test_path, transform = test_trans)
    test_data_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size)
    data_validation = torchvision.datasets.ImageFolder(validation_path, transform = test_trans)
    validation_data_loader = torch.utils.data.DataLoader(data_validation, batch_size = batch_size)
    return train_data_loader, test_data_loader, validation_data_loader
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Device being used for the run: ", device)
    logger.info('Hyperparameters are LR: ',args.lr, '; Batch Size: ', args.batch_size)
    logger.info('Data Paths: ',args.data_dir)
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_dir, args.batch_size)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model = model.to(device)

    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index = 133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info('Start Training the Model')
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info('Start Testing the Model')
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info('Start Saving the Model')
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 20, metavar = "N", help = "default: 20")
    parser.add_argument("--lr", type = float, default = 0.1, metavar = "LR", help = "default: 0.1")
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)
