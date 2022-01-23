import os
import sys
import subprocess
from flask import render_template
from flask import request
from web_app import app
from web_app.utils import *
import numpy as np
import pandas as pd


@app.route('/')
def root():
    return input()


@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/input')
def input():
    
    # purge previous image
    purge_image()
    
    # load list of images in the test set
    form_opts_path = os.path.join(os.getcwd(),'web_app/static/textdata/form_options.csv')
    form_opts = pd.read_csv(form_opts_path)
    form_opts = form_opts['img_name'].tolist()
    
    return render_template(
        "input.html",
        form_opts = form_opts
    )


@app.route('/output')
def output():
    
    # get image name 
    img_name = request.args.get('img_name')
    
    # some arguments
    endpoint_name = app.config.get('endpoint_name')
    function_name = app.config.get('lambda_function_name')
    s3_bucket = app.config.get('s3_bucket')
    s3_prefix = app.config.get('s3_prefix')
    
    # invoke lambda function to get prediction
    if True :
        img, prediction = invoke_lambda(
            endpoint_name = endpoint_name,
            function_name = function_name,
            s3_bucket = s3_bucket,
            s3_prefix = s3_prefix,
            img_name = img_name
        )
        
        # save image
        save_image(img)
    
        # predicted object count
        obj_count = np.argmax(prediction,1)[0]

    #except:
        
    #    img = 1
    #    save_image(img)
    #    obj_count = 1
    
    return render_template(
        "output.html",
        img_name = img_name,
        obj_count = obj_count
    )
    
    