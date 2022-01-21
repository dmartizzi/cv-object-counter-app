import os
import sys
import io
import copy
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import dependencies for Debugging and Profiling
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class AmazonBinImageDataset(Dataset):
    '''
    Dataset to load Amazon bin images 
    and their labels
    '''
    
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None
    ):
        '''
        Constructor
        
        Args:
            csv_file : str, path to the csv file with annotations
            
            root_dir : str, directory where the images are stored
        '''
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        '''
        Size of the dataset
        '''
        return(self.metadata.shape[0])

    def __getitem__(self,idx):
        '''
        Get image and label for given idx
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_id = self.metadata.iloc[idx,0]
        img_label = self.metadata.iloc[idx,2]
        img_name = os.path.join(self.root_dir,
                                '%05d.jpg'%img_id)
        image = Image.open(img_name)
            
        if self.transform:
            image = self.transform(image)
        
        return(image,img_label)

    
def test(
    model,
    test_loader,
    criterion,
    optimizer,
    epoch,
    device,
    hook
):
    '''
    Evaluate model on the test set
    
    Input : 
    
        model : Module, pytorch model to be trained
        
        test_loader : DataLoader, data loader
                
        criterion : _Loss, loss function
        
        optimizer : Optimizer, optimizer 
        
        epoch : int, index of this epoch

        device : device to use for computation
                
        hook : hook for the debugger
    '''
    
    model.eval()
    hook.set_mode(modes.EVAL)
    
    loss_counter=0
    running_loss = 0.0
    running_rmse = 0.0
    running_corrects = 0
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_rmse += torch.sum(torch.pow((preds-labels),2)).item()
        running_corrects += torch.sum(preds == labels).item()
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = float(running_corrects) / len(test_loader.dataset)
    epoch_rmse = (running_rmse / len(test_loader.dataset))**0.5
    
    logger.info(f'Epoch {epoch} Testing Loss: {epoch_loss}')
    logger.info(f'Epoch {epoch} Testing Accuracy: {epoch_acc}')
    logger.info(f'Epoch {epoch} Testing RMSE: {epoch_rmse}')
            
    
def train(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch,
    device,
    hook
):
    '''
    Evaluate model on the test set
    
    Input : 
    
        model : Module, pytorch model to be trained
        
        train_loader : DataLoader, data loader
                
        criterion : _Loss, loss function
        
        optimizer : Optimizer, optimizer 
        
        epoch : int, index of this epoch

        device : device to use for computation
                
        hook : hook for the debugger
    
    Output : 
    
        model : Module, trained pytorch model
    '''

    model.train()
    hook.set_mode(modes.TRAIN)
    
    loss_counter=0
    running_loss = 0.0
    running_rmse = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_rmse += torch.sum(torch.pow((preds-labels),2)).item()
        running_corrects += torch.sum(preds == labels).item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = float(running_corrects) / len(train_loader.dataset)
    epoch_rmse = (running_rmse / len(train_loader.dataset))**0.5
    
    logger.info(f'Epoch {epoch} Training Loss: {epoch_loss}')
    logger.info(f'Epoch {epoch} Training Accuracy: {epoch_acc}')
    logger.info(f'Epoch {epoch} Training RMSE: {epoch_rmse}')
            
    return(model)
    
    
def net():
    '''
    PyTorch model for multiclass 
    classification built on top of ResNet50, 
    with custom output layers, and the 
    first 6 layers frozen.
    
    Output :
        
        model : Module, pytorch model
    '''
    
    model = models.resnet50(pretrained=True)
 
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 6)
    )
    return(model)


def create_data_loaders(data, batch_size):
    '''
    Create data loaders
    
    Input : 
    
        data : str, path to data 
        
        batch_size : int, batch size
        
    Output : 
    
        train_data_loader : DataLoader, training data loader
        
        test_data_loader : DataLoader, testing data loader
        
        validation_data_loader : DataLoader, validation data loader
    '''
    
    train_mdata_path = os.path.join(data, 'train.csv')
    test_mdata_path = os.path.join(data, 'test.csv')
    validation_mdata_path=os.path.join(data, 'validation.csv')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = AmazonBinImageDataset(
        csv_file=train_mdata_path,
        root_dir=data,
        transform=train_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True
    )

    test_data = AmazonBinImageDataset(
        csv_file=test_mdata_path,
        root_dir=data,
        transform=test_transform
    )
    test_data_loader  = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True
    )

    validation_data = AmazonBinImageDataset(
        csv_file=validation_mdata_path,
        root_dir=data,
        transform=train_transform
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data, 
        batch_size=batch_size, 
        shuffle=True
    ) 
    
    return(
        train_data_loader, 
        test_data_loader, 
        validation_data_loader
    )


def main(args):
    '''
    Main function that trains and validates the model
    
    Args:
    
        args : arguments passed by the parser
    '''
    
    logger.info(f'Learning Rate: {args.learning_rate}')
    logger.info(f'Batch Size: {args.batch_size}')
    logger.info(f'Number of Epochs {args.epochs}')
        
    logger.info(f'Data Paths: {args.data}')
    
    # Data loaders
    train_loader, test_loader, validation_loader = \
        create_data_loaders(args.data, args.batch_size)
    
    # Model definition
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    model=model.to(device)
    
    # Hook for debugger
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    #Register loss for debugging
    hook.register_loss(criterion)
    
    # Train
    logger.info("Starting Model Training")
    for epoch in range(args.epochs):
        model=train(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            epoch,
            device,
            hook
        )

        test(
            model, 
            test_loader, 
            criterion, 
            optimizer, 
            epoch,
            device,
            hook
        )
    
    # Save 
    logger.info("Saving Model")
    torch.save(
        model.cpu().state_dict(), 
        os.path.join(args.model_dir, "model.pth")
    )

    
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float
    )
    parser.add_argument(
        '--batch_size', 
        type=int
    )
    parser.add_argument(
        "--epochs",
        type=int
    )
    parser.add_argument(
        '--data', type=str, 
        default=os.environ['SM_CHANNEL_TRAINING']
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--output_dir', type=str, 
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    
    args=parser.parse_args()
    
    main(args)
