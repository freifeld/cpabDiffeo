#!/usr/bin/env python
"""
Created on Thu May 29 10:25:37 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from pylab import plt
import numpy as np

import matplotlib
matplotlib.rc('text', usetex = True)

from of.gpu import CpuGpuArray
from of.utils import ObsoleteError

class Visualize(object):
    def __init__(self):
        pass
    
    
    
    
    @staticmethod
    def simple(x,v,interval,src,transformed_fwd,
                                transformed_inv=None,
               subplot_layout=[2,2],
               use_titles=True,cpa_space=None):
        """
        v is assumed to be evaluated on x. 
        So for CDFs, for example, x stands for values in the *range*
        of F, not its domain.
        """
        if cpa_space is None:
            raise ObsoleteError
        if not isinstance(x,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(v,CpuGpuArray):
            raise ObsoleteError  
        if not isinstance(src,CpuGpuArray):
            raise ObsoleteError         
        if not isinstance(transformed_fwd,CpuGpuArray):
            raise ObsoleteError  
        if not isinstance(transformed_inv,CpuGpuArray):
            raise ObsoleteError  
        
        lw = 2 
        
        xmin=cpa_space.XMINS[0]
        xmax=cpa_space.XMAXS[0]
        
        use_subplots = (subplot_layout is None) == False
        if use_subplots:
            M,N=subplot_layout
            if M*N !=4:
                raise ValueError
        if use_subplots:
            plt.subplot(M,N,1)
        else:
            plt.figure()
            
        plt.plot(x.cpu,v.cpu,lw=lw)
        if use_titles:
            plt.title('velocity')   
        ax = plt.gca()
        ax.xaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1))       
        ax.xaxis.grid(True) # For v, show only the xgrid

   
        if use_subplots:
#            print
#            print 'N',N
#            print 'M',N
#            print
            plt.subplot(M,N,2)
        else:
            plt.figure()
        plt.plot(interval,src.cpu,lw=lw)
        if use_titles:        
            plt.title('source') 
        plt.axis('scaled')
        ax = plt.gca()
        ax.xaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1)) 
        ax.yaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1)) 
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
  
        if use_subplots:
            plt.subplot(M,N,3)
        else:
            plt.figure()
        plt.plot(interval,transformed_fwd.cpu,'b',lw=lw)
        plt.plot(interval,transformed_inv.cpu,'r',lw=lw)
        if use_titles:        
            plt.title('fwd (b) \& inv (r)')
        if cpa_space.zero_v_across_bdry[0]:
            plt.axis('scaled')
        ax = plt.gca()
        ax.xaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1)) 
        
        ax.xaxis.grid(True)
#        if cpa_space.zero_v_across_bdry[0]:
        ax.yaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1)) 
        ax.yaxis.grid(True)

        if use_subplots:
            plt.subplot(M,N,4)
        else:
            plt.figure()
        ax = plt.gca()
        ax.xaxis.set_ticks(np.linspace(xmin,xmax,cpa_space.nC+1)) 
        ax.xaxis.grid(True)
        
            
#            
        dx=interval[1]-interval[0]
        # Fwd Finite Difference
        approx_derivative = np.diff(transformed_fwd.cpu.ravel())/dx
        plt.plot(interval[:-1],approx_derivative,'b',lw=lw)
        plt.ylim(0,approx_derivative.max()*1.5)

        approx_derivative = np.diff(transformed_inv.cpu.ravel())/dx
        plt.plot(interval[:-1],approx_derivative,'r',lw=lw)
        plt.ylim(0,approx_derivative.max()*1.5)       
        
        if use_titles:        
            plt.title('derivatives')
            
            
            
            
            
            
            
            
    @staticmethod
    def simple_hist(x_dense,src_dense,transformed_dense,interval,v,
     x_select,src_select,transformed_select,centers,subplot_layout=[2,2],
               use_titles=True):
        """
        v is assumed to be evaluated on x (either x_dense or x_select). 
        So for CDFs, for example, x stands for values in the *range*
        of F, not its domain.
        """
        if len(centers) != len(x_select):
            raise ValueError(centers.shape,x_select.shape)
        
        if not isinstance(x_dense,CpuGpuArray):
            raise TypeError
        if not isinstance(src_dense,CpuGpuArray):
            raise TypeError  
        if not isinstance(transformed_dense,CpuGpuArray):
            raise TypeError 
        if not isinstance(v,CpuGpuArray):
            raise TypeError   
        if not isinstance(x_select,CpuGpuArray):
            raise TypeError   
        if not isinstance(src_select,CpuGpuArray):
            raise TypeError              
        lw=2            
            
        ind = centers  # the x locations for the groups               
        width = 0.4    # the width of the bars                
                   
            
            
            
        use_subplots = (subplot_layout is None) == False
        if use_subplots:
            M,N=subplot_layout
            if M*N !=4:
                raise ValueError
        if use_subplots:
            plt.subplot(M,N,1)
        else:
            plt.figure()
            
       
            
             
        plt.plot(x_dense.cpu,v.cpu,lw=2)
        if use_titles:
            plt.title('velocity')       
   
        if use_subplots:
            plt.subplot(M,N,2)
        else:
            plt.figure()
#        plt.plot(ind,src)
        
        ax = plt.gca()
        rects1 = ax.bar(ind,x_select.cpu, width, color='r')               
        plt.xlim(0,centers.max()+1)  
        
        
        
        if use_titles:        
            plt.title('source')  
  
        if use_subplots:
            plt.subplot(M,N,3)
        else:
            plt.figure()
#        plt.plot(ind,transformed)
        
        ax = plt.gca()
        rects1 = ax.bar(ind,transformed_select.cpu, width, color='r')     
        plt.xlim(0,centers.max()+1) 

        
        if use_titles:        
            plt.title('transformed')  

        if use_subplots:
            plt.subplot(M,N,4)
        else:
            plt.figure() 
        
        h = np.hstack([transformed_select.cpu[0],
                       np.diff(transformed_select.cpu[:,0])])
        ax = plt.gca()                                 
        rects1 = ax.bar(ind,h, width, color='r')        
        plt.xlim(0,centers.max()+1) 
        
        plt.ylim(0,1)
        
       
        
        if use_titles:        
#            plt.title('histogram(="uncumsum")')            
            plt.title('histogram')      
            
            
            
            
            
            
            
            
            
            