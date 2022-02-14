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


#TODO: Import dependencies for Debugging andd Profiling

import smdebug.pytorch as smd





def test(model, test_loader, criterion, device, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    for inputs, target in test_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)
        loss = criterion(output, target)
        _, pred = torch.max(output,1)
        test_loss += loss.item() * inputs.size(0)
        
        correct += torch.sum(pred == target.data)

    total_loss = test_loss//len(test_loader)
    total_acc = correct.double()//len(test_loader) * 100

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    hook.set_mode(smd.modes.TRAIN)
    loss_count = 0
    least_loss = 99999999999
    epoch =20
    
    for e in range(epoch):
        model.train()
        hook.set_mode(smd.modes.TRAIN)
        train_loss = 0.0
        correct = 0
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(output,1)
            train_loss += loss.item() * inputs.size(0)
            correct+= torch.sum(pred == target.data)
        epoch_loss = train_loss // len(train_loader)
        epoch_accuracy = correct.double() // len(train_loader)
        logger.info('epoch{}, train loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(e,
                                                                                epoch_loss,
                                                                            epoch_accuracy,
                                                                                least_loss))
                    
        model.eval()
        hook.set_mode(smd.modes.EVAL)
        train_loss2 = 0.0
        correct2 = 0
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp, pred = torch.max(output,1)
            train_loss2 += loss.item() * inputs.size(0)
            correct2+= torch.sum(pred == target.data)
        epoch_loss = train_loss2 // len(validation_loader)
        epoch_accuracy = correct2 // len(validation_loader)
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
        if epoch == 0:
            break


    return model
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    
def net():
    model = models.resnet50(pretrained = True)
    for p in model.parameters():
        p.requires_grad = False
        nfeatures = model.fc.in_features
    model.fc=nn.Sequential(nn.Linear(nfeatures,512),
                           nn.ReLU(inplace = True),
                           nn.Linear(512,256),
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
    train_trans = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_trans = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_train = torchvision.datasets.ImageFolder(train_path, train_trans)
    train_data_loader = torch.utils.data.DataLoader(data_train, batch_size)
    data_test = torchvision.datasets.ImageFolder(test_path, test_trans)
    test_data_loader = torch.utils.data.DataLoader(data_test, batch_size)
    data_validation = torchvision.datasets.ImageFolder(validation_path, test_trans)
    validation_data_loader = torch.utils.data.DataLoader(data_validation, batch_size)
    return train_data_loader, test_data_loader, validation_data_loader
    
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device =torch.device("cpu")
    print("Device being used for the run: ", device)
    print('Hyperparameters are LR: ',args.lr, '; Batch Size: ', args.batch_size)
    print('Data Paths: ',args.data_dir)
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_dir, args.batch_size)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index = 133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook.register_loss(loss_criterion)
    
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    print('Start Training the Model')
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    print('Start Testing the Model')
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    print('Start Testing the Model')
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    print('Start Saving the Model')
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument("--batch_size", type = int, default = 20, metavar = "N", help = "default: 20")
    parser.add_argument("--lr", type = float, default = 0.1, metavar = "LR", help = "default: 0.1")
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    print(args)
    main(args)
