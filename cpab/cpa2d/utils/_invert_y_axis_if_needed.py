#!/usr/bin/env python
"""
Created on Thu Mar  6 15:53:51 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
raise Exception("I think this is obsolete. Use of.plt instead")
from pylab import plt 
def invert_y_axis_if_needed():
    g = plt.gca()
    bottom, top = g.get_ylim()    
    if top>bottom:
        g.set_ylim(top, bottom)
     
