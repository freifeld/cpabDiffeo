#!/usr/bin/env python
"""
Created on Mon Oct 13 16:29:47 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pylab import plt
from of.utils import *

from MCMC_InferenceAlgorithm import MCMC_InferenceAlgorithm
 
class Metropolis(MCMC_InferenceAlgorithm):
    def __init__(self,ll_func,proposal,lp_func=None,wlp=0.001,
                 use_ave_ll=True):
        super(type(self),self).__init__(ll_func,lp_func,use_ave_ll=use_ave_ll)
        self.proposal = proposal
        self.nSteps=0
        self.nAccepted=0
        self.level=ll_func.level
        self.wlp=wlp
        
        
    def set_theta(self,theta):
#        raise Exception("TODO: toggle btwn local and global params")
        self.d = len(theta)
        self.theta_current = theta.copy()
        self.theta_proposed = np.zeros_like(self.theta_current)
        self.ll_current = self.ll_func(self.theta_current)
        if self.use_ave_ll:             
            self.ll_current /=  self.ll_func._ll_func.nPts
            
        self.record.ll = [self.ll_current]
        if self.lp_func is not None:
            self.lp_current = self.lp_func(self.theta_current)
             
            self.record.lp = [self.lp_current]
        
        self.record.accept_ratio = []
    def step(self):
        self.nSteps+=1
        theta_current = self.theta_current
        theta_proposed = self.theta_proposed
        
        self.proposal(theta_current,theta_proposed)
#        ipshell('hi')
#        1/0
#        self.theta_proposed *= 1000 # DEBUG
        
#        if self.nSteps % 10 == 0:
#            theta_proposed/=np.linalg.norm(theta_proposed)        
        
        
        ll_proposed = self.ll_func(theta_proposed) 
        
         
        if self.use_ave_ll:              
            ll_proposed /=  self.ll_func._ll_func.nPts
        log_ratio =  ll_proposed - self.ll_current
        
        self.log_ratio=log_ratio
#        print 'll_proposed , self.ll_current:',ll_proposed , self.ll_current
#        print 'log_ratio',log_ratio
        
        lp_func = self.lp_func
        if lp_func is not None:
            lp_current = self.lp_current
            lp_proposed = lp_func(theta_proposed)
            log_prior_ratio = lp_proposed - lp_current
            log_ratio += self.wlp * log_prior_ratio
        else:
            lp_proposed=None

#        log_ratio = 1 # force accept
        
        if log_ratio>0:
            self.accept(theta_proposed,ll_proposed,lp_proposed)
        else:
            u = np.random.rand()
            if np.log(u)>log_ratio:
                self.accept(theta_proposed,ll_proposed,lp_proposed)    
        
        accept_ratio =  np.float(self.nAccepted)*100.0 / self.nSteps
        self.record.accept_ratio.append(accept_ratio)
        
#        ipshell('hi'); 1/0

    def accept(self,theta_proposed,ll_proposed,lp_proposed):
#        if self.normalize_theta:
#            if theta_proposed.any():
#                theta_proposed /= np.linalg.norm(theta_proposed)
        self.nAccepted +=1
#        ipshell('hi')
#        1/0
        self.ll_current = ll_proposed
        
        np.copyto(dst=self.theta_current,src=theta_proposed)
        self.record.ll.append(self.ll_current)
        if self.lp_func is not None:
            self.lp_current = lp_proposed          
            self.record.lp.append(self.lp_current)

    def run(self,N):
#        self.normalize_theta=normalize_theta
        
        current_run = len(self.record.runs)-1
        self.record.ll_start = self.ll_current
        if self.lp_func is not None:
            self.record.lp_start = self.lp_current
            self.record.wlp = self.wlp
        self.record.runs.append(Bunch())
        self.record.runs[current_run].theta_start = self.theta_current.copy() 
        self.record.N = N
        tic = time.clock()
        for i in range(N):
            self.step()
            if i % 200 == 0 or i == 0:
                print 'level', self.level, 'iter ',i+1 ,'/', N
                print '\tar {0}%'.format(self.record.accept_ratio[-1])
                print '\tll',self.ll_current
                if self.lp_func is not None:
                    print '\tw*lp',self.wlp*self.lp_current
#                print '\tlast log_ratio',self.log_ratio
                    
        toc = time.clock()                               
        # Now that we are done, call the ll_func again with 
        # the current theta. This may seem redundant, but it is important:
        # The current class (i.e. the sampler), can't know what ll_func is 
        # doing internally.  It is possible that ll_func modifies
        # some other things besides what it returns. For example,
        # in my transformation code, it modifies the transformed points, etc.
        # So, for safety, it is better to call it again, in case the last
        # accepted theta was not the last proposed theta.
        self.ll_func(self.theta_current)  
    
    
        

        # Record
        self.record.ll_final = self.ll_current
        if self.lp_func is not None:
            self.record.lp_final = self.lp_current
        self.record.runs[current_run].tic = tic               
        self.record.runs[current_run].toc = toc        
        self.record.runs[current_run].time = toc-tic 
        self.nRuns += 1 
        for attr in ['nSteps','nAccepted','d']:
            self.record[attr]= getattr(self,attr)
 
        self.record.runs[current_run].theta_final = self.theta_current.copy()       
#        



        
    def plot_ar(self):
        plt.plot(self.record.accept_ratio)        

if __name__ == "__main__":
    pass
