#!/usr/bin/env python
"""
Created on Fri Dec  5 10:44:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from of.utils import *
import numpy as np
import scipy
import pylab
#from pyvision.essentials import *
from pyimg import *
def get_data(name,imresize_factor=1):
    b = Bunch()
#    img = scipy.misc.lena()
#    
#    b.img1 = img.copy()
#    b.img2 = img.copy()
    
    if name=='MRI-2D-simulation_1':
        f1 = os.path.join(HOME,'data/orig/medical/2D-simulation_1','im.png')
        f2 = os.path.join(HOME,'data/orig/medical/2D-simulation_1','imr.png')
        b.img1 = pylab.imread(f1).astype(np.float)[40:-40,10:-10].copy() 
        b.img2 = pylab.imread(f2).astype(np.float)[40:-40,10:-10].copy()     
        b.img1 +=0.01*np.random.standard_normal(b.img1.shape)
        b.img2 +=0.01*np.random.standard_normal(b.img2.shape)
    elif name.startswith('MNIST'):
        parts = name.split('_')
        if len(parts)!=5:
            raise ValueError(parts)
        digit,i,junk,j= parts[1:]
        
        dname=os.path.join(HOME,'data/orig/MNIST/examples')
        f1 = os.path.join(dname,'_'.join([digit,i+'.png']))
        f2 = os.path.join(dname,'_'.join([digit,j+'.png']))
        FilesDirs.raise_if_file_does_not_exist(f1)
        FilesDirs.raise_if_file_does_not_exist(f2)
        b.img1 = pylab.imread(f1).astype(np.float)
        b.img2 = pylab.imread(f2).astype(np.float)
        
        b.img1*=255
        b.img2*=255
        # upsample

        
       
        
    else:
        raise NotImplementedError(name)
    
    b.img1 = Img(b.img1).imresize(imresize_factor)
    b.img2 = Img(b.img2).imresize(imresize_factor)
    
    return b
    

if __name__ == "__main__":
    data = get_data('MNIST_four_00009_to_00013')


    
    print data.keys()

    
    plt.figure(1)
    plt.clf()
    plt.subplot(121)
    plt.imshow(data.img1,interpolation="Nearest")#;plt.colorbar()
    plt.subplot(122)
    plt.imshow(data.img2,interpolation="Nearest")#;plt.colorbar()
    