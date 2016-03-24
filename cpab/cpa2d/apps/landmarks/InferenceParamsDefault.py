#!/usr/bin/env python
"""
Created on Thu Mar 24 12:38:01 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


class InferenceParamsDefault(object):
    def __init__(self):
        self.use_local=True
        self.vol_preserve=False
        self.MCMCniters_per_level = 10000  
        self.use_prior=True            
        self.nLevels=2  
        self.base=[2,2]     
        self.tess='I'
        self.scale_quiver = 2000
        self.valid_outside=False
        self.zero_v_across_bdry = [1,1]
        self.proposal_scale = 0.0001

if __name__ == "__main__":
    pass
