#!/usr/bin/env python
"""
Created on Wed Apr  8 15:19:52 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.utils import *
from pyvision.essentials import Img
from pylab import plt 
 
def colored_squares(dimy,dimx,nPixels_in_square_side):
    """
    """
    M=nPixels_in_square_side    
    seg = np.zeros((dimy,dimx),dtype=np.int32)  
    yy,xx = np.mgrid[:dimy,:dimx] 
    xx = xx.astype(np.float)
    yy = yy.astype(np.float)
   
    dimx = float(dimx)
    dimy=float(dimy)        
    nTimesInX = np.floor(xx / M).max() + 1
 
    seg = np.floor(yy / M)  * nTimesInX + np.floor(xx / M)
    seg = seg.astype(np.int32)
    return seg


def random_permute_labels(seg):
    p=np.random.permutation(seg.max()+1)   
    seg2 = np.zeros_like(seg)
    for c in range(seg.max()+1):             
        seg2[seg==c]=p[c]
    return seg2.astype(np.int32)


if __name__ == "__main__":  
    tic = time.clock()
    seg= colored_squares(512*2, 512*2,64*4)  
    toc = time.clock()
    print toc-tic 
    plt.figure(1)
    plt.clf()  
    plt.imshow(seg,interpolation="Nearest")
    plt.axis('scaled') 