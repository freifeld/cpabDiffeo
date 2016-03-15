#!/usr/bin/env python
"""
Created on Sun Feb  9 16:47:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import pickle, pdb
import pylab 
from pylab import plt
import numpy as np
#pylab.ion()
pylab.close('all')

#from js.utils.timing import StopWatch
from transform_cuda import CpavfCalcsGPU
from of.utils import *

dim = 2
mytype = np.float64

if dim == 1:
  #my_dict = Pkl.load('./Debug1D.pkl')
  my_dict = Pkl.load('./Debug1Dwarparound.pkl')
elif dim == 2:
  my_dict = Pkl.load('./Debug2D.pkl')
nCells = my_dict['Trels'].shape[0]

print my_dict.keys()
print my_dict['pts_at_0'].shape
N = my_dict['pts_at_0'].shape[0]
#print my_dict['CPU_results'].shape

# call to the transform function
#sw = StopWatch()
#calc_flowline_arr1d(**my_dict)
#sw.toc("flowline CPU")
#pos0 = np.zeros((my_dict['x_old'].shape[0],2))
#pos0[:,0] = my_dict['x_old'];
#pos0[:,1] = my_dict['y_old'];

#X,Y = np.meshgrid(np.arange(512),np.arange(512))
#pos0 = np.zeros((512*512,2))
#pos0[:,0] = X.ravel()
#pos0[:,1] = Y.ravel()
#
#sw.tic()
cpavf_calcs_gpu = CpavfCalcsGPU(nCells,mytype,dim)
#sw.toctic("GPU init")
#print my_dict['pts_at_0']
posT = np.zeros_like(my_dict['pts_at_0'])
posT = cpavf_calcs_gpu.calc_transformation(
  my_dict['xmins'],
  my_dict['xmaxs'],
  my_dict['Trels'],my_dict['As'],
  my_dict['pts_at_0'],my_dict['dt'],
  my_dict['nTimeSteps'],my_dict['nStepsODEsolver'],
  posT )
#sw.toctic("GPU compute")
v = cpavf_calcs_gpu.calc_velocities(
  my_dict['xmins'],
  my_dict['xmaxs'],
  my_dict['As'],
  my_dict['pts_at_0'])
#sw.toctic("GPU velocities")
posH = cpavf_calcs_gpu.calc_trajectory(
  my_dict['xmins'],
  my_dict['xmaxs'],
  my_dict['Trels'],my_dict['As'],
  my_dict['pts_at_0'],my_dict['dt'],
  my_dict['nTimeSteps'],my_dict['nStepsODEsolver'])

# to have the starting points in the history is as well
posH = np.r_[my_dict['pts_at_0'],posH]
 
# make sure the ending points are the same for both methods
T = my_dict['nTimeSteps']
if np.any((posH[(T)*N : (T+1)*N,:]-posT) > 1e-6):
  print (posH[(T)*N : (T+1)*N,:]-posT)
  raise ValueError
#pdb.set_trace()

#print posT.shape
#print posT.T

if dim == 1:
  fig=plt.figure()
  plt.subplot(5,1,1)
  plt.plot(np.arange(posT.size),my_dict['pts_at_0'][:,0],'.r')
  plt.subplot(5,1,2)
  plt.plot(np.arange(posT.size),posT[:,0],'.b',label='GPU')
  plt.legend()
  plt.subplot(5,1,3)
  #plt.plot(np.arange(posT.size),my_dict['CPU_results'][0,:],'.r',label='CPU')
  plt.plot(np.arange(posT.size),posT[:,0],'.b')
  plt.legend()
  plt.subplot(5,1,4)
  plt.plot(np.arange(v.size),v[:,0],'.b',label='GPU velocities')
  plt.legend()
  plt.subplot(5,1,5)
  for i in range(0,N,32):
#    print posH[i::N].shape
    plt.plot(np.ones(T+1)*i,posH[i::N],'r-x')
    plt.plot(np.ones(1)*i,posH[i,0],'bo')
  plt.legend()
  plt.show()
elif dim == 2:
  fig=plt.figure()
  plt.subplot(4,1,1)
  plt.plot(my_dict['pts_at_0'][:,0],my_dict['pts_at_0'][:,1],'.r')
  plt.gca().invert_yaxis()  
  plt.subplot(4,1,2)
  plt.plot(posT[:,0],posT[:,1],'.b',label='GPU')
  plt.gca().invert_yaxis()  
  plt.legend()
  plt.subplot(4,1,3)
  plt.plot(my_dict['CPU_results'][0,:],my_dict['CPU_results'][1,:],'.r',label='CPU')
  plt.plot(posT[:,0],posT[:,1],'.b',label='GPU')
  plt.gca().invert_yaxis()  
  plt.legend()
  plt.subplot(4,1,4)
  for i in range(0,N,32):
    plt.plot(posH[i::N,0],posH[i::N,1],'r')
#    plt.plot(posH[i::N,0],posH[i::N,1],'r')
#    plt.plot(posH[i,0],posH[i,1],'bo')
  plt.gca().invert_yaxis()  
  plt.legend()

  plt.figure()
#  plt.plot(v[:,0],v[:,1],'.b',label='GPU velocities')
  
  plt.quiver(my_dict['pts_at_0'][:,0][::10],
             my_dict['pts_at_0'][:,1][::10], 
            v[:,0][::10],v[:,1][::10])
  plt.gca().invert_yaxis()  
  plt.axis('scaled')
  plt.legend()
  pylab.show()
#raw_input()
 
