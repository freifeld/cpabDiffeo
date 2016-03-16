#!/usr/bin/env python
"""
Created on Fri Oct 17 15:23:03 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pylab import plt
from InferenceAlgorithm import InferenceAlgorithm
class MCMC_InferenceAlgorithm(InferenceAlgorithm):
    def plot_ll(self,**kwargs):
        """
        Plot log likelihood
        """
        plt.plot(self.record.ll,**kwargs)
    def plot_wlp(self,**kwargs):
        """
        Plot weighted log prior
        """
        if self.lp_func is None:
            return         
        plt.plot(np.asarray(self.record.lp)*self.wlp,**kwargs)
    def plot_wlp_plus_ll(self,**kwargs):
        """
        Plot log likelihood + weighted log prior
        """
        if self.lp_func is None:
            return 
        plt.plot(np.asarray(self.record.ll)+
                 np.asarray(self.record.lp)*self.wlp,**kwargs)


if __name__ == "__main__":
    pass
