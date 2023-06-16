#Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader

from torchvision.datasets import ImageFolder
import os
import sys
import json
from functools import partial


from smdebug import modes
import smdebug.pytorch as smd
from smdebug.pytorch import get_hook

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, hook):
    '''
        This function take a model and a 
          testing data loader and get the test accuray/loss of the model
          It also include debugging/profiling hooks that might be needed
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss=0
    correct=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()

    test_loss /= len(test_loader.dataset)
    
    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, hook, epochs):
    '''
        This function take a model and
          data loaders for training and get the train model
          It also include debugging/profiling hooks that might be needed
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    loss_counter=0
  
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples=0
    
        for step, (inputs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
      
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples+=len(inputs)
          
            if running_samples>(0.2*len(train_loader.dataset)):
                break

        epoch_loss = running_loss / running_samples
        epoch_acc = running_corrects / running_samples
              
        logger.info(f"Epoch {epoch}: Loss {epoch_loss}, Accuracy {100*epoch_acc}%")
    
    return model
    
def net():
    '''
        This function initializes a pretrained model
    '''
    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, 5),
                nn.LogSoftmax(dim=1)
    )
    
    return model


def create_data_loaders(data, batch_size):
    '''
    This function is use to create data loaders
    '''
    logger.info("Create the data loader")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),#
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),#
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_set = ImageFolder(root=os.path.join(data, 'train'), transform=train_transform)
    test_set = ImageFolder(root=os.path.join(data, 'test'), transform=val_transform)
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    
    return {'train_loader': train_loader, 'test_loader': test_loader}


def main(args):
    '''
        Initialize a model by calling the net function
    '''
    model=net()
    
    hook = get_hook(create_if_not_exists=True)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
#     hook = smd.Hook(out_dir=args.model_dir)
    
#     hook.register_module(model)
    
    
    '''
        Create your loss and optimizer
    '''
    #loss_criterion = nn.NLLLoss()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #hook.register_loss(loss_criterion)
    
    '''
        Call the train function to start training the modelon the training data from S3
    '''
    logger.info(f"Hyperparameter values: Batch size {args.batch_size}: Number of epochs {args.epochs}, Learning rate {args.lr}")
    data =  create_data_loaders(args.data_dir, args.batch_size)
    model=train(model, data['train_loader'], loss_criterion, optimizer, hook,args.epochs)
    
    '''
        Test the model to see its accuracy
    '''
    test(model, data['test_loader'], loss_criterion, hook)
    
    '''
        Save the trained model
    '''
    logger.info("Saving the model.")
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
        Specify all the hyperparameters that is needed to use to train the model.
    '''
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar="N",
        help="Input batch size for the training (default:32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 50)",
    )
    
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    
    main(args)
