#!/usr/bin/env python
"""
Created on Sat Oct 18 11:07:04 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from scipy.special import erf

def cdf_1d_gaussian(x,mu=0,sigma=1):
        return 0.5*(1+erf((x-mu)/np.sqrt(2)/sigma))  

if __name__ == "__main__":
    pass















