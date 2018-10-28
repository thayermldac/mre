## Well commented dual objective experiment
## Second objective using simplified viscoelastic equation
# Basics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random

# Keras
import keras
import keras.backend as K
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Lambda, UpSampling2D, BatchNormalization
from keras.layers import Dot, dot, add, multiply

# Others
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%matplotlib inline 

df=pd.read_msgpack('../MREdata_102418.msg')
#column includes: filename, freq, fslice, RS(\mu), Ui,Ur


# Prepare Data
Ur = np.stack(df.Ur.values,axis=0)
Ui = np.stack(df.Ui.values,axis=0)
Freq=df.Freq.values #values are np.array 
x_data = np.sqrt(Ui**2+Ur**2) #numpy default is elementwise operation

y_data = np.stack(df.RS.values,axis=2).transpose(2,0,1) #Rs is mu
y_data = y_data/10000.

#right side of viscoelastic equation:
#-\omega*\rho*u
#first replicate freq (\omega) to N*64*64*3 to multiply elemently
omega2=(2*3.14*Freq)**2 #omega squared
omega2=omega2.reshape([Freq.size,1,1,1]);
omega2=np.tile(omega2,[1,64,64,3]) #np.tile is equivalent to Matlab reshape
rho=1 #predefined \rho
aux_data=rho*omega2  #aux_data has frequency feature (expanded to correct dimension) shape: N*64*64*3





print('X size:', x_data.shape)
print('Y size:', y_data.shape)


# Split to Train & Valid
x_train, x_test, aux_train,aux_test, y_train, y_test = train_test_split(x_data,aux_data,y_data,test_size=0.3)
x_train, x_valid,aux_train,aux_valid, y_train, y_valid = train_test_split(x_train,aux_train,y_train,test_size=0.3)

print('Training examples:   ', len(x_train))
print('Validation examples: ', len(x_valid))
print('Testing examples:    ', len(x_test))

#define laplace using tensorflow
def laplace(x):
    # Make Kernel
    a = np.asarray([[0., 1., 0.],
                    [1.,-4., 1.],
                    [0., 1., 0.]])    #a is kernal of doing second derivative

    a = a.reshape(list(a.shape) + [1,1]) #expand a to 4-D
    kernel = K.constant(a,dtype=1)
    
    # Do Convolution
    x = K.expand_dims(K.expand_dims(x, 0), -1) #insert 1 dimension at the beginnig and end of x
    y = K.depthwise_conv2d(x,kernel, padding='same') # do conv2d: second derivative
    
    return y[0,:,:,0]

def laplacian(x): #doing laplacian on the norm of u
    u = K.tf.norm(x,axis=3) 
    return K.map_fn(laplace,u) 


def laplacian3D(x):
    u1 = K.map_fn(laplace,x[:,:,:,0]) #x direction
    u2 = K.map_fn(laplace,x[:,:,:,1]) #y direction
    u3 = K.map_fn(laplace,x[:,:,:,2]) #z direction 
#    u  = K.tf.add_n([u1,u2,u3])
    u  = K.tf.stack([u1,u2,u3],axis=3)#laplacian of u (u is a vector)
    return u


def muStack(y):
    return K.tf.stack([y,y,y],axis=3)

U=np.sqrt(Ur*Ur+Ui*Ui)

# Parameters
xshp   = x_train.shape[1:]
nbatch = 64
nepoch = 3

# Architecture
L1 = 64
L2 = 50
L3 = 32
L4 = 32

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

## recreate the CNN with residual layers
#Autoencoder


# Build Neural Model
# residual neural network with autoencoding
# Encoding
x  = Input(shape=xshp,name='Input')
aux = Input(shape=xshp,name='aux_input')
x1=BatchNormalization()(x)
#h  = Conv2D(L1,kernel_size=(5,5),strides=(2,2),activation='relu',padding='same',name='E1')(x1)
h=residual_block(x1,L1,(2,2))
#h  = Conv2D(L2,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',name='E2')(h)
h=residual_block(h,L2,(2,2))
#h  = Conv2D(L3,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',name='E3')(h)
h=residual_block(h,L3,(2,2))
#e  = Conv2D(L4,kernel_size=(2,2),strides=(1,1),activation='relu',padding='same',name='E4')(h)
e=residual_block(h,L4,(1,1))
# Decoding
#h  = Conv2D(L4,kernel_size=(2,2),activation='relu',padding='same',name='D1')(e)
h=residual_block(e,L4)
h  = UpSampling2D((2,2))(h)
#h  = Conv2D(L3,kernel_size=(3,3),activation='relu',padding='same',name='D2')(h)
h=residual_block(h,L3)
h  = UpSampling2D((2,2))(h)
#h  = Conv2D(L2,kernel_size=(3,3),activation='relu',padding='same',name='D3')(h)
h=residual_block(h,L2)
h  = UpSampling2D((2,2))(h)
#h  = Conv2D(1,kernel_size=(5,5),activation='relu',padding='same',name='D4')(h)
h=residual_block(h,L1)
#h=residual_block(x1,64,(2,2))
#h=residual_block(x,50, (2,2))
#h=residual_block(x,32,(2,2))
#h=UpSampling2D((2,2))(x)
#h=residual_block(x,32)
#h=UpSampling2D((2,2))(x)
#h=residual_block(x,32)
#h=UpSampling2D((2,2))(x)
#h=residual_block(x,1,(1,1),True)


h1=BatchNormalization()(h)
y  = Lambda(lambda xx: K.squeeze(xx,3),name='Recon')(h1)

# Laplacian
print(x.shape)
#l1= Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,0],-1))
#l2 = Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,1],-1))
#l3 = Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,2],-1))

#l=add([l1,l2,l3])

l = Lambda(laplacian3D,name='Laplacian')(x)
m = Lambda(muStack,name='StackedMu')(y)

#z = dot([y,l],axes=[1,2],name='Mre')
equal = multiply([m,l],name='equal') # \mu*laplacian(u) %left of viscoelastic equation
equar = multiply([aux,x],name='equlr') # rho*w^2*u
equa = add([equal,equar],name='Mre') # viscoelastic equation

# z = dot([y,l],axes=-1,name='Mre')

# Build Model
#model = Model(inputs=x,outputs=y)
#model.summary()

# Build Aux Model
aux   = Model(inputs=[x,aux],outputs=[y,equa])
aux.summary()

# Compiling Model
#model.compile(loss='mse',optimizer='adam')
aux.compile(loss='mse',loss_weights=[1.,0], optimizer='adam')


#x_aux = np.linalg.norm(x_train,axis=-1)



# Train Model
# this is the dual-objective part. Obj1 is y_train ,obj2 is viscoelastic equation (should be 0)
log = aux.fit([x_train,aux_train],[y_train,np.zeros_like(aux_train)],
             epochs=nepoch,
             batch_size=nbatch)


y_pred = aux.predict([x_test,aux_test])
y_pred1=[y]
# Visualize Examples
nimg=10
fig = plt.figure(figsize=(10,6))
for i in range(nimg):
    ax=plt.subplot(3,nimg,i+1)
    plt.imshow(y_test[i+10])
    ax=plt.subplot(3,nimg,i+1+nimg)
    plt.imshow(y_pred[0][i+10])
    ax=plt.subplot(3,nimg,i+1+nimg+nimg)
    plt.imshow(y_pred[1][i+10])
    
fig.savefig('plotfile.png') 
    

fig = plt.figure(figsize=(10,6))
def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    historydf.to_pickle('history.pkl')
    #historydf = pd.read_pickle('history.pkl')
    plt.figure(figsize=(8, 6))
    historydf['loss'].plot(ylim=(0, historydf['loss'].values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
    
plot_loss(log)
fig.savefig('plossfile.png') 
