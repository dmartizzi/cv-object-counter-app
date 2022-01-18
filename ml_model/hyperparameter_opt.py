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
    device
):
    '''
    Evaluate model on the test set
    
    Input : 
    
        model : Module, pytorch model
        
        test_loader : DataLoader, data loader
        
        criterion : _Loss, loss function
        
        device : device to use for computation
    '''
    
    model.eval()
    running_loss=0.0
    running_rmse=0.0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_rmse += torch.sum(torch.pow((preds-labels),2))
        running_corrects += torch.sum(preds == labels)

    total_loss = running_loss // len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    total_rmse = (running_rmse // len(test_loader))**0.5
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    logger.info(f"Testing RMSE: {total_rmse}")

    
def train(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    epochs,
    device
):
    '''
    Evaluate model on the test set
    
    Input : 
    
        model : Module, pytorch model to be trained
        
        train_loader : DataLoader, data loader
        
        validation_loader : DataLoader, data loader
        
        criterion : _Loss, loss function
        
        optimizer : Optimizer, optimizer 
        
        epochs : int, number of epochs
        
        device : device to use for computation
    
    Output : 
    
        model : Module, trained pytorch model
        
    '''
    
    best_loss=1e6
    image_dataset={'Training':train_loader, 'Validation':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['Training', 'Validation']:
            if phase=='Training':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_rmse = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='Training':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_rmse += torch.sum(torch.pow((preds-labels),2))
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase])
            epoch_rmse = (running_rmse // len(image_dataset[phase]))**0.5
            
            if phase=='Validation':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

            logger.info(f'Epoch {epoch} {phase} Loss: {epoch_loss}')
            logger.info(f'Epoch {epoch} {phase} Accuracy: {epoch_acc}')
            logger.info(f'Epoch {epoch} {phase} RMSE: {epoch_rmse}')
            logger.info(f'Epoch {epoch} Best Loss: {best_loss}')
            
        if loss_counter==1:
            break
        if epoch==0:
            break
            
    return(model)
    
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 10))
    return(model)


def create_data_loaders(data, batch_size):
    '''
    
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
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Train
    logger.info("Starting Model Training")
    model=train(
        model, 
        train_loader, 
        validation_loader, 
        criterion, 
        optimizer, 
        args.epochs,
        device
    )
    
    # Test
    logger.info("Testing Model")
    test(
        model, 
        test_loader, 
        criterion, 
        device
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
