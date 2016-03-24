#!/usr/bin/env python
"""
Created on Wed May  7 11:30:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from _ScaleDependentLogPriorGeneral import ScaleDependentLogPriorGeneral
#from of.utils import ipshell
#eps = 1e-16

class ScaleDependentLogPrior(ScaleDependentLogPriorGeneral):     
    def calc_lp(self,theta):
        """
        Calculate log prior
        """
        J = self.cpa_cov_inv
        mu = self.mu
        return -0.5 * (theta-mu).dot(J).dot(theta-mu)