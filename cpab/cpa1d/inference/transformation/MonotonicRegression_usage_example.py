#!/usr/bin/env python
"""
Created on Sat Oct 18 13:58:15 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np

from of.utils import *

from cpab.cpa1d.inference.transformation.MonotonicRegression import MonotonicRegression



if __name__ == "__main__":
    import pylab
    from pylab import plt
    import of.plt
    if not inside_spyder():
        pylab.ion() 
    from cpab.distributions.standard_cdfs import cdf_1d_gaussian
    
    np.random.seed(0)  
    x = np.linspace(-10,10,1000*10)
    
    x = x[::10]
    
    # Create some increasing function
    y = cdf_1d_gaussian(x,mu=-4,sigma=1)  
    y *=0.3
    y += 0.7*cdf_1d_gaussian(x,mu=4,sigma=2)    
    y *=10     
    y +=3
    
    range_start=y.min()
    range_end=y.max()
    
    # Add noise
    y += 0.4*np.random.standard_normal(y.shape)
    
    
    if 1:
        plt.figure(0)
        of.plt.set_figure_size_and_location(1000,0,1000,500)
        plt.clf()
        plt.subplot(121)
        plt.cla()
        plt.plot(x,y,'.',lw=3)
        plt.title('data')
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=50)
        ax.tick_params(axis='x', labelsize=30)
        
         
    
     

    
    nPtsDense = 10000
    mr = MonotonicRegression(base=[12],nLevels=4)    
    mr.set_dense(domain_start=-10,domain_end=10)                    
    mr.set_data(x=x,y=y,range_start=range_start,range_end=range_end)
     
    print mr
    
    if 1:
        plt.figure(0)
        plt.subplot(122)
        plt.cla()
        mr.plot_dst('.r')  
        mr.plot_src('.g')  
        plt.legend(['dst','src'],'lower right')
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=50)
        ax.tick_params(axis='x', labelsize=30)        

        
    mr.set_run_lengths([500,500,500,50000,50000][:mr.nLevels])
    
    theta,inference_record = mr.fit(use_prior=1)
    plt.figure(1000)
    of.plt.set_figure_size_and_location(50,50,1000,1000)
    plt.clf()
    mr.plot_inference_summary(inference_record)    
    


    if not inside_spyder():
        raw_input('raw_input:')
