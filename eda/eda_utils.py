import os
import sys
import json
import boto3
import numpy as np
import pandas as pd
from PIL import Image


def create_image_subsample(n_full=500000,\
                           n_sel=50000):
    '''
    Random shuffle a list of ids and select 
    a subsample
    
    Input :
        image_ids_sel_path : str, path were ids are stored
    
        n_full : int, size of the full set
        
        n_sel : int, size of the desired subsample
        
    Output : 
        image_ids_sel : ndarray, the desired subsample ids     
    '''
    if n_sel > n_full:
        print('Warning! n_full < n_sel! Setting n_full = n_sel!')
    
    image_ids_full = np.arange(1,n_full+1)
    np.random.shuffle(image_ids_full)
    image_ids_sel = image_ids_full[:n_sel]
    
    return(image_ids_sel)


def load_image_ids_sel(image_ids_sel_path='image_ids_sel.csv'):
    '''
    Loads randomly selected subsample of image ids
    
    Input :
        image_ids_sel_path : str, path were ids are stored
    
        n_full : int, size of the full set
        
        n_sel : int, size of the desired subsample
        
    Output :
        
        image_ids_sel : DataFrame, the selected image ids
    '''
    
    image_ids_sel = pd.read_csv(image_ids_sel_path)
    n_sel = image_ids_sel.shape[0]
    print(f'Loaded random subsample of {n_sel} images.')
    
    return(image_ids_sel)


def generate_image_ids_sel(image_ids_sel_path='image_ids_sel.csv',\
                           n_full=500000,\
                           n_sel=50000):
    '''
    Generates randomly selected subsample of image ids
    
    Input :
        image_ids_sel_path : str, path were ids are stored
    
        n_full : int, size of the full set
        
        n_sel : int, size of the desired subsample
        
    Output :
        
        image_ids_sel : DataFrame, the selected image ids
    '''
    
    image_ids_sel = pd.DataFrame(data={
        'id' : create_image_subsample(n_full,n_sel)
    })
    image_ids_sel.sort_values(by=['id'],ascending=True,\
                              ignore_index=False,\
                              inplace=True)
    image_ids_sel.to_csv(image_ids_sel_path,index=False)
    print(f'Generated random subsample of {n_sel} images out of {n_full}.')
        
    return(image_ids_sel)


def get_image_item_count(idd):
    '''
    Retrieve item count for a given image id 
    Input :
        idd : int, id of the image
        
    Output :
        count : int, number of items in the image
    '''
    client = boto3.client('s3')
    response = client.get_object(Bucket='amazon-bin-images',\
                                 Key='%05d.json'%idd)
    body = json.loads(response['Body'].read())
    count = body['EXPECTED_QUANTITY']
    
    return(count)


def get_image(idd):
    '''
    Retrieve an image for a given id
    
    Input : 
        idd : int, id of the image
        
    Output : 
        img : ImageFile, image object
    '''
    client = boto3.client('s3')
    response = client.get_object(Bucket='amazon-bin-images',\
                                 Key='%05d.jpg'%idd)
    img = Image.open(response['Body'])
    
    return(img)

    
def generate_labels(count,thr):
    '''
    Define labels for training
    
    Input : 
        count : int, number of items in the image
        
    Output : 
        label : int, label for training
    '''
    if count < thr:
        label = count
    else:
        label = thr
    
    return(label)
    
    
def train_val_test_split(image_ids_sel,\
                         split=(0.8,0.1,0.1)):
    '''
    Perform a random train, validation, test split
    
    Input : 
        image_ids_sel : DataFrame, image metadata
        
        split : tuple, (train,val,test)
    Output : 
        train_df : DataFrame, training images metadata
        
        val_df : DataFrame, validation images metadata
        
        test_df : DataFrame, testing images metadata
    '''
    
    full_df = image_ids_sel.copy()\
                           .sample(frac=1.0,random_state=1294)\
                           .reset_index(drop=True)
    
    n_full = full_df.shape[0]
    n_train_ini = 0
    n_val_ini = int(split[0]*n_full)+1
    n_test_ini = int((split[0]+split[1])*n_full)+1
    
    cols = full_df.columns.tolist()
    train_df = pd.DataFrame(columns=cols,dtype=int)
    val_df = pd.DataFrame(columns=cols,dtype=int)
    test_df = pd.DataFrame(columns=cols,dtype=int)
    
    if n_full > 0:
        if n_val_ini-n_train_ini > 0:
            train_df = full_df.iloc[0:n_val_ini,:]\
                              .sort_values(by=['id'],\
                                           ascending=True,\
                                           ignore_index=True)
            
        if n_test_ini-n_val_ini > 0:
            val_df = full_df.iloc[n_val_ini:n_test_ini,:]\
                            .sort_values(by=['id'],\
                                         ascending=True,\
                                         ignore_index=True)
        
        if n_full-n_test_ini > 0:
            test_df = full_df.iloc[n_test_ini:n_full,:]\
                             .sort_values(by=['id'],\
                                          ascending=True,\
                                          ignore_index=True)
            
    return(train_df,val_df,test_df)


def transfer_images(idd,s3source,s3dest):
    '''
    Move image idd.jpg from source 
    to destination
    
    Input :
    
        idd : int, id of the image
        
        s3source : str, URI of source 
        
        s3dest : str, URI of destination
    '''
    
    client = boto3.resource('s3')
    key = '%05d.jpg'%idd
    copy_source = {
      'Bucket': s3source,
      'Key': key
    }
    bucket = client.Bucket(s3dest)
    bucket.copy(copy_source, key)
    