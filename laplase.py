#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:46:22 2018

@author: syp
"""

def laplace(x):
    # Make Kernel
    a = np.asarray([[0., 1., 0.],
                    [1.,-4., 1.],
                    [0., 1., 0.]])    
#     a = np.asarray([[0.5, 1.0, 0.5],
#                     [1.0, -6., 1.0],
#                     [0.5, 1.0, 0.5]])
    a = a.reshape(list(a.shape) + [1,1])
    kernel = K.constant(a,dtype=1)
    
    # Do Convolution
    x = K.expand_dims(K.expand_dims(x, 0), -1)
    y = K.depthwise_conv2d(x,kernel, padding='same')
    
    return y[0,:,:,0]

def laplacian(x):
    u = K.tf.norm(x,axis=3)
    return K.map_fn(laplace,u)

import pandas as pd
import numpy as np
import keras.backend as K

df=pd.read_msgpack('/Users/syp/Desktop/git/mre/MREdata_072118.msg')
Ur=np.stack(df.Ur.values,axis=3).transpose(3,0,1,2)
Ui=np.stack(df.Ui.values,axis=3).transpose(3,0,1,2)

X=np.sqrt(Ui**2+Ur**2)

Y=np.stack(df.RS.values,axis=2).transpose(2,0,1)
#Y=Y.reshape(612,-1)
Y.ptp()
laplacian(X)