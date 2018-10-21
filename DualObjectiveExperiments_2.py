## Well commented dual objective experiment
## Second objective using simplified viscoelastic equation
# Basics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Keras
import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Lambda, UpSampling2D
from keras.layers import Dot, dot, add, multiply

# Others
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%matplotlib inline 

df=pd.read_msgpack('MREdata_080618.msg')
#column includes: filename, freq, fslice, RS(\mu), Ui,Ur


# Prepare Data
Ur = np.stack(df.Ur.values,axis=0)
Ui = np.stack(df.Ui.values,axis=0)
Freq=df.Freq.values #values are np.array 
x_data = np.sqrt(Ui**2+Ur**2) #numpy default is elementwise operation

y_data = np.stack(df.RS.values,axis=2).transpose(2,0,1) #Rs is mu
y_data = y_data/10000.

print('X size:', x_data.shape)
print('Y size:', y_data.shape)


# Split to Train & Valid
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3)
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=0.3)

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
nbatch = 16
nepoch = 100

# Architecture
L1 = 64
L2 = 50
L3 = 32
L4 = 32

# Build Neural Model
# residual neural network with autoencoding
# Encoding
x  = Input(shape=xshp,name='Input')
h  = Conv2D(L1,kernel_size=(5,5),strides=(2,2),activation='relu',padding='same',name='E1')(x)
h  = Conv2D(L2,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',name='E2')(h)
h  = Conv2D(L3,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',name='E3')(h)
e  = Conv2D(L4,kernel_size=(2,2),strides=(1,1),activation='relu',padding='same',name='E4')(h)

# Decoding
h  = Conv2D(L4,kernel_size=(2,2),activation='relu',padding='same',name='D1')(e)
h  = UpSampling2D((2,2))(h)
h  = Conv2D(L3,kernel_size=(3,3),activation='relu',padding='same',name='D2')(h)
h  = UpSampling2D((2,2))(h)
h  = Conv2D(L2,kernel_size=(3,3),activation='relu',padding='same',name='D3')(h)
h  = UpSampling2D((2,2))(h)
h  = Conv2D(1,kernel_size=(5,5),activation='relu',padding='same',name='D4')(h)
y  = Lambda(lambda xx: K.squeeze(xx,3),name='Recon')(h)

# Laplacian
print(x.shape)
#l1= Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,0],-1))
#l2 = Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,1],-1))
#l3 = Lambda(laplacian,name='Laplacian')(K.tf.expand_dims(x[:,:,:,2],-1))

#l=add([l1,l2,l3])

l = Lambda(laplacian3D,name='Laplacian')(x)
m = Lambda(muStack,name='StackedMu')(y)

#z = dot([y,l],axes=[1,2],name='Mre')
z = multiply([m,l],name='Mre') #\mu*laplacian(u)
# z = dot([y,l],axes=-1,name='Mre')

# Build Model
#model = Model(inputs=x,outputs=y)
#model.summary()

# Build Aux Model
aux   = Model(inputs=x,outputs=[y,z])
aux.summary()

# Compiling Model
#model.compile(loss='mse',optimizer='adam')
aux.compile(loss='mse',  optimizer='adam')


#x_aux = np.linalg.norm(x_train,axis=-1)

#right side of viscoelastic equation:
#-\omega*\rho*u
#first replicate freq (\omega) to N*64*64*3 to multiply elemently
omega=2*3.14*Freq.reshape([Freq.size,1,1,1])
omega=np.tile(omega,[1,64,64,3]) #np.tile is equivalent to Matlab reshape
rho=1000 #predefined \rho
obj2=-rho*omega*U

# Train Model
# this is the dual-objective part. Obj1 is y_train 
log = aux.fit(x_train,[y_train,obj2],
             epochs=nepoch,
             batch_size=nbatch)


y_pred = aux.predict(x_test)
y_pred1=[y]
# Visualize Examples
nimg=3
for i in range(nimg):
    ax=plt.subplot(3,nimg,i+1)
    plt.imshow(y_test[i+10])
    ax=plt.subplot(3,nimg,i+1+nimg)
    plt.imshow(y_pred[0][i+10])
    ax=plt.subplot(3,nimg,i+1+nimg+nimg)
    plt.imshow(y_pred[1][i+10])   
    
    
def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
    


plot_loss(log)

