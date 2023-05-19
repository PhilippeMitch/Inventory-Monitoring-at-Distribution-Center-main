# Code reference
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint
# https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html?highlight=invoke#making-predictions-with-the-aws-cli
# https://github.com/aws/sagemaker-tensorflow-serving-container#prepost-processing
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html

import os
from PIL import Image
import io
import json
import pickle
import numpy as np
import torchvision.models as models
import torch.nn as nn
from sagemaker_inference import content_types, decoder
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from PIL import ImageFile
from PIL import ImageEnhance
import requests

import sys
import logging
import argparse
import os
from io import BytesIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):

    logger.info("In model_fn. Model directory is - ", model_dir)
    
    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, 5),
                nn.LogSoftmax(dim=1)
    )
    
    logger.info("Load the model from, ", model_dir)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return model


def input_fn(request_body, content_type='application/json'):
    
    if content_type == 'application/json':
        logger.info(f'The request body {request_body}')
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        
        image_data = Image.open(requests.get(url, stream=True).raw)
        enhancer_object = ImageEnhance.Contrast(image_data)
        out = enhancer_object.enhance(1.6)
        enhancer_object = ImageEnhance.Sharpness(out)
        image_data = enhancer_object.enhance(1.7)
        test_valid_transform = transforms.Compose([
                                     transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info('Process the image')
        img_tensor = test_valid_transform(image_data)
        return img_tensor
    
    raise Exception(f'ContentType is not supported {content_type}')



def predict_fn(input_data, model):
    
    input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
        
    return ps


def output_fn(prediction_output, accept='application/json'):
    
    logger.info('List of classes')
    
    classes = {
        0: '1', 
        1: '2', 
        2: '3', 
        3: '4', 
        4: '5'
        }
    

    topk, topclass = prediction_output.topk(3, dim=1)
    result = []
    
    for i in range(3):
        pred = {'Prediction': classes[topclass.numpy()[0][i]], 'Score': f'{topk.numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    
    raise Exception(f'ContenteType is not supported:{accept}')