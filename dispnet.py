#!/usr/bin/env python3

from __future__ import print_function

import sys
import os
import time
import copy

import numpy as np
import theano
import theano.tensor as T
#from theano.compile.nanguardmode import NanGuardMode

import lasagne

from pfm import *

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

import os.path
import glob

import random

import math

DISP_SCALE = 160
#DISP_SCALE = 1
DEBUG_SCALE = 1

TRAINING_WIDTH = 960
TRAINING_HEIGHT = 540


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

def freeze(*layers):

    for layer in layers :
        for param in layer.params :
            layer.params[param].discard('trainable')
    return #layer  # optional, if you want to use it in-line

def build_cnn(input_var=None, freeze_inner_layers = False):
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

    if freeze_inner_layers :
        freeze(conv2,conv3a,conv3b,conv4a,conv4b,conv5a,conv5b,conv6a,conv6b,pr6,upconv5,upscale5,merge5,iconv5,pr5
               ,upconv4,upscale4,merge4,iconv5,pr4,upconv3,upscale3,merge3,iconv3,pr3,upconv2,upscale2,merge2,iconv2,pr2)

    return (pr1,pr2,pr3,pr4,pr5,pr6)

def resize_expand(img, size):
    result = Image.new(img.mode, size)
    resized = img.copy()

    ratio = min(size[0]/img.size[0], size[1]/img.size[1])
    scaled_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))

    #resized.thumbnail(size)
    resized = resized.resize(scaled_size, resample=Image.BILINEAR) #TODO BICUBIC??

    offset = (math.floor((size[0] - resized.size[0])/2),math.floor((size[1]-resized.size[1])/2))
    result.paste(resized, offset)
    return result

def make_batch(left, right, disp=None, disp_masks=None):
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

        masks = np.empty([batchsize,1,height//2,width//2], dtype=np.float32)

        disp_factors = []
    
        for i in range(0,batchsize):
            disp_factor = TRAINING_HEIGHT / left[i].size[1]
            disp_factor /= DISP_SCALE
            disp_factors.append(1.0/disp_factor)

            inputs[i,:3] = np.array(resize_expand(left[i],(width,height))).astype(np.float32).transpose(2,0,1)[:3] / 255.0
            inputs[i,3:] = np.array(resize_expand(right[i],(width,height))).astype(np.float32).transpose(2,0,1)[:3] / 255.0
    
            if(disp == None): continue

            for target_idx in range(0,6):
                scale = 2**(target_idx+1)
                w = int(width/scale)
                h = int(height/scale)
                targets[target_idx][i] = np.array(resize_expand(disp[i],(w,h))).astype(np.float32).reshape(1,h,w) * disp_factor

            if(disp_masks == None):
                masks[i].fill(1)
            else:
                masks[i] = np.array(resize_expand(disp_masks[i],(int(width/2),int(height/2)))).astype(np.float32).reshape(1,int(height/2),int(width/2))
                masks[i][masks[i] > 0] = 1
    
        return inputs, targets, masks, disp_factors

def augment_batch(batch):
    inputs = copy.deepcopy(batch[0])
    outputs = copy.deepcopy(batch[1])
    masks = copy.deepcopy(batch[2])
    disp_factors = copy.deepcopy(batch[3])

    for inp in inputs:
        r = random.uniform(0.9,1.1)
        g = random.uniform(0.9,1.1)
        b = random.uniform(0.9,1.1)
        left_r = r * random.uniform(0.9,1.1)
        left_g = g * random.uniform(0.9,1.1)
        left_b = b * random.uniform(0.9,1.1)
        right_r = r * random.uniform(0.9,1.1)
        right_g = g * random.uniform(0.9,1.1)
        right_b = b * random.uniform(0.9,1.1)

        inp[0] *= left_r
        inp[1] *= left_g
        inp[2] *= left_b
        inp[3] *= right_r 
        inp[4] *= right_g
        inp[5] *= right_b

    return (inputs,outputs,masks,disp_factors)


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


    def load_training_batch(self,offset, batchsize=10):
        offset *= batchsize
        batchsize = min(batchsize,len(train_disps)-offset)
        left = [Image.open(path) for path in train_left[offset:offset+batchsize]]
        right = [Image.open(path) for path in train_right[offset:offset+batchsize]]
        disp = [load_pfm(path) for path in train_disps[offset:offset+batchsize]]
        return make_batch(left,right,disp)

    def load_validation_batch(self,offset, batchsize=10):
        offset *= batchsize
        batchsize = min(batchsize,len(val_disps)-offset)
        left = [Image.open(path) for path in val_left[offset:offset+batchsize]]
        right = [Image.open(path) for path in val_right[offset:offset+batchsize]]
        disp = [load_pfm(path) for path in val_disps[offset:offset+batchsize]]
        return make_batch(left,right,disp)


class DispNet( object):

    def __init__(self, freeze_inner_layers = False ):
        self.build(freeze_inner_layers)

    def build(self, freeze_inner_layers):
        print("Building model and compiling functions...")

        self.weight_var = theano.shared(np.array([0,0,0,0,0,1],dtype=np.float32),'weights')
        self.learning_rate_var = theano.shared(np.array(0.001,dtype=np.float32),'learning_rate')

        self.input_var = T.tensor4('input')
        self.pixel_mask = T.tensor4('pixel mask')
        self.prediction_layers = build_cnn(self.input_var, freeze_inner_layers)
        self.network = self.prediction_layers[0]
    
        self.outputs = lasagne.layers.get_output(self.prediction_layers)
        #self.outputs = [lasagne.layers.get_output(layer) for layer in self.prediction_layers]
        self.prediction = self.outputs[0]

        self.target_vars = [T.tensor4(("target{}").format(i+1)) for i in range(0,6)]
        #self.losses = [T.mean(lasagne.objectives.squared_error(pred,var))
        #        for pred,var in zip(self.outputs,self.target_vars)]

        self.losses = [abs(pred-var).mean()
                for pred,var in zip(self.outputs,self.target_vars)]
        self.loss = lasagne.objectives.aggregate(T.stack(self.losses), weights = self.weight_var, mode='normalized_sum')


        self.outer_loss = (self.pixel_mask * abs(self.outputs[0]-self.target_vars[0])).sum() / self.pixel_mask.sum()

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

        self.updates = lasagne.updates.adam(
            self.loss, self.params, 
            learning_rate=self.learning_rate_var,
            beta1 = 0.9,
            beta2 = 0.999 )

        self.outer_updates = lasagne.updates.adam(
            self.outer_loss, self.params,
            learning_rate=self.learning_rate_var,
            beta1 = 0.9,
            beta2 = 0.999 )

        '''
        self.updates = lasagne.updates.nesterov_momentum(
                self.loss, 
                self.params, 
                self.learning_rate_var)
        '''

        self.train_outer_fn = theano.function(
                name="train_outer",
                inputs=[self.input_var, self.pixel_mask, self.target_vars[0]],
                outputs=self.outer_loss,
                updates=self.outer_updates,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )

        self.train_fn = theano.function(
                inputs=[self.input_var] + self.target_vars,
                outputs=self.loss,
                updates=self.updates,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
        self.validate_fn = theano.function(
                inputs=[self.input_var] + self.target_vars,
                outputs=self.loss,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
        self.validate_outer_fn = theano.function(
                inputs=[self.input_var, self.pixel_mask, self.target_vars[0]],
                outputs=self.outer_loss,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
        self.predict_fn = theano.function([self.input_var], self.prediction,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
        self.debug_fn = theano.function([self.input_var], self.outputs,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )

    def set_weights(self, weights):
        weights = np.array(weights, dtype=np.float32)
        self.weight_var.set_value(weights)

    def set_learning_rate(self, learning_rate):
        self.learning_rate_var.set_value(np.array(learning_rate,dtype=np.float32))
    
    def train(self,batch):
        return self.train_fn(batch[0], *batch[1] ) * batch[3][0]

    def validate_outer(self,batch):
        return self.validate_outer_fn(batch[0], batch[2], batch[1][0]) * batch[3][0]

    def train_outer(self,batch):
        return self.train_outer_fn(batch[0], batch[2], batch[1][0]) * batch[3][0]

    def validate(self,batch):
        return self.validate_fn(batch[0], *batch[1]) * batch[3][0]

    def predict_data(self,batch_or_left_path,right_path=None):
        batch = batch_or_left_path
        if(right_path != None):
            batch = make_batch([batch_or_left_path], [right_path])

        disp = self.predict_fn(batch[0])[0]*batch[3][0]
        disp = disp.reshape(disp.shape[1:])
        return disp

    def predict_image(self,batch_or_left_path,right_path=None):
        disp = self.predict_data(batch_or_left_path,right_path).clip(0,255)
        return Image.fromarray(disp.astype(np.uint8))

    def debug(self,batch,idx=0):
        if(idx < 0 or idx >= len(batch[0])): return None

        preds = self.debug_fn(batch[0])

        grid_image = Image.new('RGB',(384*2,192*7))

        images = [Image.fromarray((np.clip(pred[idx]*batch[3][idx]*DEBUG_SCALE,0,255)).astype(np.uint8).reshape(pred[idx].shape[1:])) for pred in preds]
        images = [image.resize((384,192)) for image in images]

        for i in range(0,6):
            grid_image.paste(images[i], (0,(6-i)*192))

        left = (batch[0][idx][:3] * 255).astype(np.uint8).transpose(1,2,0)
        left = Image.fromarray(left).resize((384,192))
        grid_image.paste(left, (0,0))

        right = (batch[0][idx][3:] * 255).astype(np.uint8).transpose(1,2,0)
        right = Image.fromarray(right).resize((384,192))
        grid_image.paste(right, (384,0))

        images = [Image.fromarray((np.clip(pred[idx]*batch[3][idx]*DEBUG_SCALE,0,255)).astype(np.uint8).reshape(pred[idx].shape[1:])) for pred in batch[1]]
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
