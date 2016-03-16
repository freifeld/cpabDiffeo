#!/usr/bin/env python
"""
Created on Thu Oct 16 14:32:11 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ipshell
class LogPrior(object):
    def __init__(self,ms,msp,level,SDLP,
                 required):
        """
        SDLP is short for ScaleDependentLogPrior 
        """
#        ipshell('hopa'                        )
#        1/0
        self.level=level   
        self._lp_func = SDLP(ms,msp,level=self.level,
                           **required)   
            
    def __call__(self,alpha):
        return self._lp_func(alpha)


if __name__ == "__main__":
    pass















