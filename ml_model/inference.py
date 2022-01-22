import io
import base64
import requests
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
IMAGE_CONTENT_TYPE = 'image/*'
JPEG_CONTENT_TYPE = 'image/jpeg'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net():
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


def model_fn(model_dir):
    '''
    Retrieve model
    '''
    
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    
    return(model)


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    '''
    Get input data
    '''
    
    logger.info('Deserializing the input data.')
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    if content_type == JPEG_CONTENT_TYPE or \
      content_type == IMAGE_CONTENT_TYPE: 
        logger.debug('Loaded JPEG content')
        return(Image.open(io.BytesIO(request_body)))

    # process a URL submitted to the endpoint    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        img_content = base64.b64decode(request['image'].encode('ASCII'))

        return(Image.open(io.BytesIO(img_content)))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

    
def predict_fn(input_object, model):
    '''
    Perform inference
    '''
    
    logger.info('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

