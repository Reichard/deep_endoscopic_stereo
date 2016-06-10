#!/usr/bin/env python3

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from pfm import *

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

import os.path
import glob

import random

DISP_SCALE = 160.0

train_disps = []
train_left = []
train_right = []
val_disps = []  
val_left = []
val_right = []

dataset_root_dir = ''

def build_conv( input_layer, num_filters, filter_size, stride):
    return lasagne.layers.Conv2DLayer( input_layer, num_filters=num_filters, 
            filter_size = (filter_size,filter_size),
            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
            stride = stride, pad = 'same')

def build_upconv( input_layer, num_filters, filter_size):
    return lasagne.layers.TransposedConv2DLayer(
            input_layer, num_filters=num_filters, filter_size = (filter_size,filter_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            stride = 2,
            crop = 1)

def build_cnn(input_var=None):
    input_layer = lasagne.layers.InputLayer( shape = (None, 6, 384, 768), input_var = input_var )

    conv1 = build_conv(input_layer,64,7,2)
    
    conv2 = build_conv(conv1,128,5,2)

    conv3a = build_conv(conv2,256,5,2)
    conv3b = build_conv(conv3a,256,3,1)

    conv4a = build_conv(conv3b,512,3,2)
    conv4b = build_conv(conv4a,512,3,1)

    conv5a = build_conv(conv4b,512,3,2)
    conv5b = build_conv(conv5a,512,3,1)

    conv6a = build_conv(conv5b,1024,3,2)
    conv6b = build_conv(conv6a,1024,3,1)
    pr6 = build_conv(conv6b,1,3,1)

    upconv5 = build_upconv(conv6b,512,4)
    upscale5 = lasagne.layers.Upscale2DLayer(pr6,2)
    merge5 = lasagne.layers.ConcatLayer([upconv5,upscale5,conv5b])
    iconv5 = build_conv(merge5, 512, 3, 1)
    pr5 = build_conv(iconv5,1,3,1)

    upconv4 = build_upconv(iconv5,256,4)
    upscale4 = lasagne.layers.Upscale2DLayer(pr5,2)
    merge4 = lasagne.layers.ConcatLayer([upconv4,upscale4,conv4b] )
    iconv4 = build_conv(merge4, 256, 3, 1)
    pr4 = build_conv(iconv4,1,3,1)

    upconv3 = build_upconv(iconv4,128,4)
    upscale3 = lasagne.layers.Upscale2DLayer(pr4,2)
    merge3 = lasagne.layers.ConcatLayer( [upconv3,upscale3,conv3b] )
    iconv3 = build_conv(merge3, 128,3,1)
    pr3 = build_conv(iconv3,1,3,1)

    upconv2 = build_upconv(iconv3,64,4)
    upscale2 = lasagne.layers.Upscale2DLayer(pr3,2)
    merge2 = lasagne.layers.ConcatLayer([upconv2, upscale2, conv2] )
    iconv2 = build_conv(merge2,64,3,1)
    pr2 = build_conv(iconv2,1,3,1)

    upconv1 = build_upconv(iconv2,32,4)
    upscale1 = lasagne.layers.Upscale2DLayer(pr2,2)
    merge1 = lasagne.layers.ConcatLayer([upconv1,upscale1,conv1] )
    iconv1 = build_conv(merge1,32,3,1)
    pr1 = build_conv(iconv1,1,3,1)

    return (pr1,pr2,pr3,pr4,pr5,pr6)



class Dataset( object ):

    def __init__( self, root='/org/share/MediData/MedData/Simluation/dispnet' ):
        global train_disps
        global train_left
        global train_right
        global val_disps
        global val_left
        global val_right
        global dataset_root_dir
        dataset_root_dir = root

        disp_folder = 'disparity'
        image_folder = 'finalpass/frames_finalpass'
        train_folder = 'TRAIN'
        validation_folder = 'TEST'

        train_disps = sorted(glob.glob('{}/{}/{}/*/*/left/*.pfm'.format( root, disp_folder, train_folder )))
        train_left = sorted(glob.glob('{}/{}/{}/*/*/left/*.png'.format( root, image_folder, train_folder )))
        train_right = sorted(glob.glob('{}/{}/{}/*/*/right/*.png'.format( root, image_folder, train_folder )))

        assert(len(train_disps) == len(train_left) == len(train_right))

        shuffled = list(range(len(train_disps)))
        random.shuffle(shuffled)
        train_disps = [train_disps[i] for i in shuffled]
        train_left = [train_left[i] for i in shuffled]
        train_right= [train_right[i] for i in shuffled]

        for d,l,r in zip(train_disps,train_left,train_right):
            d_name = d.split('/')[-1].split('.')[0]
            l_name = l.split('/')[-1].split('.')[0]
            r_name = r.split('/')[-1].split('.')[0]
            assert( len(d_name) > 0 and d_name == l_name == r_name)

        val_disps = sorted(glob.glob('{}/{}/{}/*/*/left/*.pfm'.format( root, disp_folder, validation_folder )))
        val_left = sorted(glob.glob('{}/{}/{}/*/*/left/*.png'.format( root, image_folder, validation_folder)))
        val_right = sorted(glob.glob('{}/{}/{}/*/*/right/*.png'.format( root, image_folder, validation_folder)))

        self.size = len(train_disps)

    def make_batch(self, left, right, disp=None):
        assert(len(left) == len(right))
        assert(disp == None or len(disp) == len(left))
    
        batchsize = len(left)
        width = 768
        height = 384
    
        inputs = np.empty([batchsize,6,height,width], dtype=np.float32)
        targets = [
            np.empty([batchsize,1,int(height/2),int(width/2)], dtype=np.float32),
            np.empty([batchsize,1,int(height/4),int(width/4)], dtype=np.float32),
            np.empty([batchsize,1,int(height/8),int(width/8)], dtype=np.float32),
            np.empty([batchsize,1,int(height/16),int(width/16)], dtype=np.float32),
            np.empty([batchsize,1,int(height/32),int(width/32)], dtype=np.float32),
            np.empty([batchsize,1,int(height/64),int(width/63)], dtype=np.float32) ]
    
        for i in range(0,batchsize):
            if len(left) <= i: break
    
            inputs[i,:3] = np.array(left[i].resize((width,height))).astype(np.float32).transpose(2,0,1)[:3] / 255.0
            inputs[i,3:] = np.array(right[i].resize((width,height))).astype(np.float32).transpose(2,0,1)[:3] / 255.0
    
            if(disp == None): continue
    
            for target_idx in range(0,6):
                scale = 2**(target_idx+1)
                w = int(width/scale)
                h = int(height/scale)
                targets[target_idx][i] = np.array(disp[i].resize((w,h))).astype(np.float32).reshape(1,h,w) /DISP_SCALE 
    
        return inputs, targets

    def load_training_batch(self,offset, batchsize=10):
        offset *= batchsize
        batchsize = min(batchsize,len(train_disps)-offset)
        left = [Image.open(path) for path in train_left[offset:offset+batchsize]]
        right = [Image.open(path) for path in train_right[offset:offset+batchsize]]
        disp = [load_pfm(path) for path in train_disps[offset:offset+batchsize]]
        return self.make_batch(left,right,disp)

    def load_validation_batch(self,offset, batchsize=10):
        offset *= batchsize
        batchsize = min(batchsize,len(val_disps)-offset)
        left = [Image.open(path) for path in val_left[offset:offset+batchsize]]
        right = [Image.open(path) for path in val_right[offset:offset+batchsize]]
        disp = [load_pfm(path) for path in val_disps[offset:offset+batchsize]]
        return self.make_batch(left,right,disp)


class DispNet( object ):

    def __init__(self):
        self.build()

    def build(self):
        print("Building model and compiling functions...")

        #self.weight_var = theano.shared(np.array([1,1,1,1,1,1],dtype=np.float32),'weights')
        #self.learning_rate_var = theano.shared(np.array(0.001,dtype=np.float32),'learning_rate')

        #self.set_weights((1,1,1,1,1,1))
        #self.set_learning_rate(0.001)

        self.input_var = T.tensor4('input')
        self.target_vars = [T.tensor4(("target{}").format(i+1)) for i in range(0,6)]
    
        self.prediction_layers = build_cnn(self.input_var)
        self.network = self.prediction_layers[0]
    
        self.predictions = lasagne.layers.get_output(self.prediction_layers)

        self.losses = [lasagne.objectives.squared_error(pred,var).mean() 
                for pred,var in zip(self.predictions,self.target_vars)]

        #self.losses = [(abs(pred-var)).mean() 
        #        for pred,var in zip(predictions,self.target_vars)]

        #loss = lasagne.objectives.aggregate(T.stack(self.losses), weights=np.array([1,1,1,1,1,1],dtype=np.float32))
        self.loss = lasagne.objectives.aggregate(T.stack(self.losses), weights = (1,1,1,1,1,1))

        self.params = lasagne.layers.get_all_params(self.prediction_layers[0], trainable=True)
        self.updates = lasagne.updates.adam(
            self.loss, self.params, 
            #learning_rate=self.learning_rate_var,
            learning_rate = 0.001,
            beta1 = 0.9,
            beta2 = 0.999 )

        self.train_fn = theano.function([self.input_var] + self.target_vars, self.loss, updates=self.updates)

        self.validate_fn = theano.function([self.input_var] + self.target_vars, self.loss)

        self.predict_fn = theano.function([self.input_var], self.predictions[0])
        self.debug_fn = theano.function([self.input_var], self.predictions)

    def set_weights(self, weights):
        weights = np.array(weights, dtype=np.float32)
        #self.loss_weights = weights / np.linalg.norm(weights)
        self.weight_var.set_value(weights / np.linalg.norm(weights))

    def set_learning_rate(self, learning_rate):
        self.learning_rate_var.set_value(np.array(learning_rate,dtype=np.float32))
    
    def train(self,batch):
        #return self.train_fn(batch[0], self.loss_weights, self.learning_rate, *batch[1])
        return self.train_fn(batch[0], *batch[1])

    def validate(self,batch):
        return self.validate_fn(batch[0], *batch[1])

    def predict(self,batch):
        disp = (self.predict_fn(batch[0])[0]*DISP_SCALE).astype(np.uint8)
        disp = disp.reshape(disp.shape[1:])
        return Image.fromarray(disp)

    def debug(self,batch,idx=0):
        if(idx < 0 or idx >= len(batch[0])): return None

        grid_image = Image.new('RGB',(384*2,192*7))

        preds = self.debug_fn(batch[0])
        images = [Image.fromarray((np.clip(pred[idx]*DISP_SCALE,0,255)).astype(np.uint8).reshape(pred[0].shape[1:])) for pred in preds]
        images = [image.resize((384,192)) for image in images]

        for i in range(0,6):
            grid_image.paste(images[i], (0,(6-i)*192))

        left = (batch[0][idx][:3] * 255).astype(np.uint8).transpose(1,2,0)
        left = Image.fromarray(left).resize((384,192))
        grid_image.paste(left, (0,0))

        right = (batch[0][idx][3:] * 255).astype(np.uint8).transpose(1,2,0)
        right = Image.fromarray(right).resize((384,192))
        grid_image.paste(right, (384,0))

        images = [Image.fromarray((np.clip(pred[idx]*DISP_SCALE,0,255)).astype(np.uint8).reshape(pred[0].shape[1:])) for pred in batch[1]]
        images = [image.resize((384,192)) for image in images]
        for i in range(0,6):
            grid_image.paste(images[i], (384,(6-i)*192))

        return grid_image
        
    def save(self,path):
        np.savez(path, *lasagne.layers.get_all_param_values(self.network))
        
    def load(self,path):
        with np.load(path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
