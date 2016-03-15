#!/usr/bin/env python
"""
Created on Sat May 24 18:35:42 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from pylab import plt
import of.plt

class ConfigPlt(object):
    def __init__(self,Nx,Ny):
        self.Nx=Nx
        self.Ny=Ny
    def __call__(self,axis_on_or_off='off',use_lims=True,use_lims_ext=False):
        
        if 0:
            axis_on_or_off = axis_on_or_off.lower()
            if axis_on_or_off not in ['off','on']:
                raise ValueError(axis_on_or_off)
            if use_lims_ext:
                use_lims=False
            if use_lims_ext and use_lims:            
                msg="use_lims={0} AND ".format(use_lims)
                msg+="use_lims_ext={0} ".format(use_lims_ext)
                msg+="but at most one of these can be True."
                raise ValueError(msg)
            Nx=self.Nx
            Ny=self.Ny
#            plt.axis(axis_on_or_off)
            plt.axis('scaled')
            
#            if use_lims:
#                plt.xlim([0,Nx])
#                plt.ylim([0,Ny])
#            if use_lims_ext:
#                plt.xlim([0,Nx+1])
#                plt.ylim([0,Ny+1])
    
                 
            of.plt.axis_ij()  
            return
        
        axis_on_or_off = axis_on_or_off.lower()
        if axis_on_or_off not in ['off','on']:
            raise ValueError(axis_on_or_off)
        if use_lims_ext:
            use_lims=False
        if use_lims_ext and use_lims:            
            msg="use_lims={0} AND ".format(use_lims)
            msg+="use_lims_ext={0} ".format(use_lims_ext)
            msg+="but at most one of these can be True."
            raise ValueError(msg)
        Nx=self.Nx
        Ny=self.Ny
        plt.axis(axis_on_or_off)
        plt.axis('scaled')
        
        if use_lims:
            plt.xlim([0,Nx])
            plt.ylim([0,Ny])
        if use_lims_ext:
            plt.xlim([0,Nx+1])
            plt.ylim([0,Ny+1])

             
        of.plt.axis_ij()  