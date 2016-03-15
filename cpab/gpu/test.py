#!/usr/bin/env python
"""
Created on Sun Feb  9 16:47:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
 
from of.utils import *

import pylab 
from pylab import plt
import numpy as np
#pylab.ion()
pylab.close('all')


#from js.utils.timing import StopWatch
from transform_cpu.cy.transform.transform import calc_flowline_arr1d
from transform_cuda import FlowlineGPU

class TF:
    show=True
#for i in range(4):  
for i in [1]:    
    print
    nC = (2 ** i)**2
    my_dict = Pkl.load('./input_{}.pkl'.format(nC))
    
    dt = my_dict['dt']

    #print my_dict.keys()
    
    # call to the transform function
    #sw = StopWatch()
    if 0:
      calc_flowline_arr1d(**my_dict)
    #sw.toc("flowline CPU")
    #pos0 = np.zeros((my_dict['x_old'].shape[0],2))
    #pos0[:,0] = my_dict['x_old'];
    #pos0[:,1] = my_dict['y_old'];
    
    X,Y = np.meshgrid(np.arange(512),np.arange(512))
    d = 2 ** 1
    X=X[::d,::d]
    Y=Y[::d,::d]
    nPts = X.size 
    print "Cells: {} x {}".format(int(np.sqrt(nC)),int(np.sqrt(nC)))
    print "#Pts = {}  ".format(nPts)
    nStepsODEsolver= my_dict['nStepsODEsolver'] 
    for nStepsODEsolver in [10]:#[5,10,50,100]:
        print "#StepsODEsolver =",nStepsODEsolver
        pos0 = np.zeros((nPts,2))
        # This is nPts x 2
        pos0[:,0] = X.ravel()
        pos0[:,1] = Y.ravel()
        
        #sw.tic()
        flowL = FlowlineGPU(nC)
        #sw.toctic("GPU init")
        
        tic=time.clock()
#        for k in range(10):
        posT = flowL.calc_flowline(
          my_dict['xmins'],my_dict['ymins'],
          my_dict['xmaxs'],my_dict['ymaxs'],
          my_dict['Trels'],my_dict['As'],
          pos0,
          dt,
          my_dict['nTimeSteps'],nStepsODEsolver)
        toc=time.clock()
        print 'Time:',toc-tic , '[sec]'
        #sw.toctic("GPU compute")
#        ipshell('hi')
        
        
        if TF.show:
            
            fig=plt.figure()
            plt.plot(posT[:,0],posT[:,1],'.')
            plt.gca().invert_yaxis()  
            fig.show()
            
            # plotting
            
            lines_shape = (512,  512)
            
            fig=plt.figure()
            for i in range(lines_shape[0]):
              plt.plot(posT[i*lines_shape[1]:(i+1)*lines_shape[1],0],
                posT[i*lines_shape[1]:(i+1)*lines_shape[1],1])
            plt.gca().invert_yaxis()  
            fig.show()

            raw_input()
            
            hx,hy = my_dict['history_x'],my_dict['history_y']
            
            lines_shape = (18,  512)
            # The initial points
            lines_old_x=hx[0].reshape(lines_shape ).copy()
            lines_old_y=hy[0].reshape(lines_shape ).copy()
            # The final points
            lines_new_x=hx[-1].reshape(lines_shape ).copy()
            lines_new_y=hy[-1].reshape(lines_shape ).copy()
            
            c = 'r'
            fig = plt.figure()
            plt.subplot(121)
            for line_x,line_y in zip(lines_old_x,lines_old_y):
                plt.plot(line_x,line_y,c)  
                plt.axis('scaled')
                q=100
                plt.xlim(0-q,512+q)
                plt.ylim(0-q,512+q)           
                plt.gca().invert_yaxis()      
            c = 'b'
            plt.subplot(122)
            for line_x,line_y in zip(lines_new_x,lines_new_y):
                plt.plot(line_x,line_y,c)                        
                plt.axis('scaled')
                q=500
                plt.xlim(0-q,512+q)
                plt.ylim(0-q,512+q)           
                plt.gca().invert_yaxis()  
                
            pylab.show()    
