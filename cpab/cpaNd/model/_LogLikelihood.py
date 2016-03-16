#!/usr/bin/env python
"""
Created on Thu Oct 16 14:32:11 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ipshell
class LogLikelihood(object):
    def __init__(self,ms,level,SDLL,
#                 sigma,
#                 src,dst,
#                 transformed,                 
#                 params_flow_int,
                 data,
                 required):
        """
        SDLL is short for ScaleDependentLogLikelihood 
        """
#        ipshell('hopa'                        )
        self.level=level  
#        print required.keys()
#        ipshell('hi')
#        1/0
        self._ll_func = SDLL(ms=ms,level=self.level,
                           data=data,**required)   
            
    def __call__(self,alpha):
        return self._ll_func(alpha)


if __name__ == "__main__":
    pass















