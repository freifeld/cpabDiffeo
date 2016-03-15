#!/usr/bin/env python
"""
Created on Thu Mar 13 16:37:00 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ObsoleteError
raise ObsoleteError()
from of.utils import ipshell
def cpa_simple_mean(cpa_space):
#    _As=cpa_space.Avees2As(cpa_space.zeros_no_con())      
#    _mu =  cpa_space.project(cpa_space.As2Avees(_As))    

    _mu = cpa_space.get_zeros_theta()    
    return _mu      
     
