import os
import sys
import subprocess
import io
import base64
import json
import numpy as np
import pandas as pd
from PIL import Image
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel,PyTorchPredictor
from sagemaker.predictor import Predictor

class ImagePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(ImagePredictor, self).__init__(
            endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=jpeg_serializer,
            deserializer=json_deserializer,
        )
        

def invoke_lambda(
    endpoint_name = "",
    function_name = "serve-cv-object-counter",
    s3_bucket = "amazon-bin-images-sub",
    s3_prefix = "",
    img_name = "00134.jpg"):
    
    jpeg_serializer = sagemaker.serializers.IdentitySerializer("image/jpeg")
    json_deserializer = sagemaker.deserializers.JSONDeserializer()
        
    lambda_client = boto3.client('lambda')
    request = json.dumps({
        "s3_bucket" : s3_bucket,
        "s3_prefix" : s3_prefix,
        "endpoint_name" : endpoint_name,
        "image_name" : img_name
    })
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType = "RequestResponse",
        Payload=request
    )    
    output = json.loads(response['Payload'].read())['body']
        
    img = Image.open(io.BytesIO(base64.b64decode(output['image'].encode('ASCII'))))
    prediction = output['prediction']
    
    return(img,prediction)


def purge_image():
    
    img_path = os.path.join(os.getcwd(),'web_app/static/images/sel_img.jpg',)
    if os.path.exists(img_path):
        subprocess.run(["rm",img_path])
        

def save_image(img):
    
    base_path = os.path.join(os.getcwd(),'web_app/static/images')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    img_name = "sel_img.jpg"
    img_path = os.path.join(base_path,img_name)
    
    if type(img) == int:
        subprocess.run(["touch",img_path])
    else :
        img.save(img_path)