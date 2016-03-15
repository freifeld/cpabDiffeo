#!/usr/bin/env python
"""
Created on Sun Feb  2 15:58:54 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


#from of.utils import ipshell
class AffineFlow(object):           
    def __init__(self,A,verts):
#        ipshell('hi')
#        def unpack(verts):
#            return verts[:,0].min(),verts[:,0].max(),verts[:,1].min(),verts[:,1].max()
	 
  
#        xmin,xmax,ymin,ymax = unpack(verts)

        self.xmins = verts[:,:-1].min(axis=0)
        self.xmaxs = verts[:,:-1].max(axis=0)
  
        self.A=A.copy()
#        self.xmin=xmin 
#        self.xmax=xmax
 
        
        self.Trel = dict()

