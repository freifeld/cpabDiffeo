#!/usr/bin/env python
"""
Created on Wed Mar 23 10:36:27 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import inspect
import numpy as np
from of.utils import *


dirname_of_this_file = os.path.dirname(os.path.abspath(
                        inspect.getfile(inspect.currentframe())))
if len(dirname_of_this_file)==0:
    raise ValueError
dirname_of_this_file = os.path.abspath(dirname_of_this_file)

def get_data(name):
    data = Bunch()

    data.kind='landmarks'
    data.landmarks_are_lin_ordered=0    
    
    name = name.lower()
    if not name.startswith('lfw'):
        raise ValueError(name)
    parts = name.split('_')
    if len(parts)!=4:
        raise ValueError(name)
    junk1,i,junk2,j=parts    
    fname = os.path.join(dirname_of_this_file,'example_data','LFW',
    'landmarks_{}_and_{}.npz'.format(i,j))
    FilesDirs.raise_if_file_does_not_exist(fname)
    _data=np.load(fname)
    arr = _data['arr_0']
    """
    Data:
    Array of size (2, 136). 
    If we call the array arr, then arr[0] are the landmarks of image 1, 
    while arr[1] are those of image 2. 
    Each landmark vector contains 68 x-coordinates followed by 68 y-coordinates.
    """
    src = arr[0].reshape(2,-1).T 
    dst = arr[1].reshape(2,-1).T
    
    data.src=np.require(src,requirements=['C'])
    data.dst=np.require(dst,requirements=['C'])
    
    data.dname=os.path.abspath(os.path.dirname(fname))
    data.fname=fname
    
    return data


if __name__ == "__main__":
    
    from pylab import plt
    import of.plt
    name = 'LFW_5_to_6'
    data = get_data('LFW_5_to_6')
    src = data.src
    dst = data.dst
    plt.close('all')
    plt.figure(1)
    plt.plot(src[:,0],src[:,1],'go')
    plt.plot(dst[:,0],dst[:,1],'bo')
    of.plt.axis_ij()
    plt.axis('scaled')
    plt.legend(['src','dst'])