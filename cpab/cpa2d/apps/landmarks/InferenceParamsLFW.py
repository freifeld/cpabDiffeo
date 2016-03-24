#!/usr/bin/env python
"""
Created on Thu Mar 24 12:48:48 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from InferenceParamsDefault import InferenceParamsDefault
class InferenceParams(InferenceParamsDefault):
    def __init__(self):
        super(type(self),self).__init__()   
                    
        self.scale_spatial=1
        self.scale_value=1000*5
        if any(self.zero_v_across_bdry):
            self.proposal_scale = 0.01
        
        
        self.base = [8,8]; self.nLevels = 1; self.MCMCniters_per_level = 10000
        self.scale_quiver=1000
        self.sigma_lm=10


if __name__ == "__main__":
    pass
