#!/usr/bin/env python
"""
Created on Fri Mar 27 10:12:24 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import *

class ClosedFormInt(object):
    def __init__(self,tw):
#        self.tw=tw
        self.cpa_space = cpa_space = tw.ms.L_cpa_space[0]
        self.nC = cpa_space.nC
        self.x_tess = cpa_space.local_stuff.vert_tess[:,0]
        
        self.As = cpa_space.pat.As_sq_mats

        
    def calc_cell_idx(self,x):
        nC = self.nC
        idx = np.floor(x * nC).astype(np.int)    
        idx = np.minimum(nC-1,np.maximum(idx,0))
        return idx

    def calc_v(self,x,cell_idx):  
        """
        Assumes zero bdry cond as well as 0 outside.
        """   
        nC = self.nC
        As = self.As 
        
        if np.isscalar(x):
            if not 0<= cell_idx < nC:
                raise ValueError
        else:
            if not (np.logical_and(0<= cell_idx,cell_idx < nC).all()):
                raise ValueError
            
       
         
        
        if np.isscalar(x):
            if not np.isscalar(cell_idx):
                raise ValueError   
            if not 0<=x<=1:
                return 0    
             
            if As[cell_idx].shape != (2,2):
                raise ValueError
            a,b = As[cell_idx][0]                   
            return a*x + b
        else:
            if np.logical_or(x<0,x>max_val).any():
                raise NotImplementedError
            
            raise Exception("need to debug")
            
            return (As[cell_idx][:,0])*x + As[cell_idx][:,1]

    def calc_phi(self,x,velTess,t):
        nC = self.nC
        As = self.As  
        x_tess = self.x_tess
        
        if np.isscalar(x)==False:
            raise NotImplementedError(x.shape)
        x_old = x
        counter = 0
        t_reminder = t
        
        if 0<x_old<1 and x_old in x_tess:
            v = velTess[x_tess==x_old][0]
            if v>0:
                pass
            else:
#                raise ValueError(x_old,v)
                x_old = x_old-1e-16
        while True:
            
            cell_idx = self.calc_cell_idx(x_old)
            
             
            t_cell,cell_idx_next = self.find_crossing_time(x_old,cell_idx,As[cell_idx])
            if t_cell < 0:
                raise ValueError('wtf?')
            if t_cell == 0.0:
                 
                v =  self.calc_v(x_old,cell_idx)
                if v<0:
                   x_new = x_tess[cell_idx]
#                   x_new-= 1e-16 
                   
                else:
                   x_new = x_tess[cell_idx_next] 
                   
#                raise ValueError(x_old,t_cell,x_tess[cell_idx],
#                                 cell_idx,calc_v(x_old,cell_idx),counter)
                    
            if t_cell>=t_reminder:
                # We won't leave this cell
                t_cell = t_reminder
                bdry_crossed = False
                
            else:
                bdry_crossed = True
                
            



            if bdry_crossed:
                if cell_idx_next>cell_idx: # i.e. v>0
#                    # Due to numerical issues, the equality won't be perfect.
#                    if np.allclose(x_new,x_tess[cell_idx_next])==False:                
#                        raise ValueError(cell_idx,cell_idx_next,x_new,x_tess[cell_idx_next])
#                    else:
#                        x_new = x_tess[cell_idx_next] # Forcing the correct value
                   
                   
                   # right endppint of the *current* interval, and then some.
                   x_new = x_tess[cell_idx_next]
                   x_new+=1e-12
                else:
                   # Left endppint of the *current* interval, and then some.
                   x_new = x_tess[cell_idx]
                   x_new-=1e-12
            else:
                x_new = self.calc_affine_phi(x_old,t_cell,As[cell_idx])
                
            x_new = np.minimum(1.0,np.maximum(0.0,x_new))
            if not 0<=x_new<=1:               
                raise ValueError(x,x_old,x_new,t_cell,bdry_crossed,counter) 
           
            
            t_reminder -= t_cell
            
            verbose = 0
            if verbose:
                print
                print 'cell_idx:',cell_idx
                if bdry_crossed:
                    print 'cell_idx_next:',cell_idx_next
                print 't_cell:',t_cell
                print 'x_old:',x_old
                print 'x_new:',x_new
            if not bdry_crossed:                
                break
            
            counter += 1
            
            if counter >= nC:
#                1/0
#                break # due to numerics, sometimes t_reminder is not 0 even thought it should be
                raise ValueError(x_old,x_new,x_old==x_new,v,cell_idx,cell_idx_next,
                                 't_cell',t_cell,
                                 't_reminder',t_reminder,'t',t)
                
                
            # prepare for next iteration
            x_old = x_new # for vectors, we ill need to do copy
        return x_new 

    def calc_phi_multiple_pts(self,x,velTess,pts_fwd,t):
        nPts = len(x)
        for i in range(nPts):
            pts_fwd[i] =  self.calc_phi(x[i],velTess,t=t)    
    def calc_affine_phi(self,x,t,A):
        """
        Assumes zero bdry cond as well as 0 outside.
        """   
        if np.isnan(t):
            raise ValueError
        if np.isscalar(x)==False:
            raise NotImplementedError    
        if A.shape != (2,2):
            raise ValueError(A.shape)
        a,b = A[0,0],A[0,1] 

        if not 0<=x<=1:
            a,b=0,0
        
        if a == 0:
            x_new =  x + t * b
        else:
            x_new =  np.exp(t*a)*x + b*(np.exp(t*a)-1)/a
        if np.isnan(x_new):
            raise ValueError(x,t,A)
        return x_new 
    def find_crossing_time(self,x,cell_idx,A):
        nC = self.nC
        x_tess = self.x_tess
        v = self.calc_v(x,cell_idx)
        if np.isscalar(x)==False:
            raise NotImplementedError
        if np.isscalar(cell_idx)==False:
            raise NotImplementedError 
        if A.shape != (2,2):
            raise ValueError(A.shape)
        a,b = A[0]

        if np.isscalar(a)==False:
                raise ValueError(a.shape) 
        if np.isscalar(b)==False:
                raise ValueError(b.shape) 
        
        if np.allclose(v,0):
            return np.inf,None
        
        cell_idx_next = cell_idx + np.int(np.sign(v))
        if  not 0<= cell_idx_next < nC:
            return np.inf,None
        if v>0:
            x_new = x_tess[cell_idx_next] # pick the right endpoint
            x_new+=1e-16
        else:
            x_new = x_tess[cell_idx] # pick the left endpoint
            x_new-=1e-16
            
        
        v_new =  self.calc_v(x_new,cell_idx_next)
        if np.allclose(v_new,0):
            # We will never make it! I.e., time = \infty
            return np.inf,None
         
        if np.sign(v)!=np.sign(v_new):
            # We will never make it! I.e., time = \infty
            return np.inf,None
            
    
     
        if x_new==0 or x_new == 1:
            t=np.inf
        elif np.allclose(a, 0):
            t =  (x_new - x)/b
        else:
            ratio =  (x_new + b/a)  / (x + b/a)
            log_ratio = np.log(ratio)
            t = 1.0/a * np.log(     (x_new + b/a) /
                                    (x     + b/a)  )
                    
                           
        if np.isscalar(t)==False:
            raise ValueError(t.shape)
        if np.isnan(t):
            raise ValueError('a,b',a,b,'x','x_new',x,x_new,'v',v)
        if t<0:
            
#            t=1e-16
            
            raise ValueError(x,x_new,a,b,a>0,v,np.allclose(x,x_new), 
                             ratio,log_ratio,t==0 )
        return t,cell_idx_next    
               

if __name__ == "__main__":
    from cpa.cpa1d.TransformWrapper import TransformWrapper

    nC = 5        
    nPtsDense = 1000   
    
    tw = TransformWrapper(nCols=100,
                          nLevels=1,  
                          base=[nC],
                          nPtsDense=nPtsDense)
    closed_form_int = ClosedFormInt(tw=tw)   
    
    
    