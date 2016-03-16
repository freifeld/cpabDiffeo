#!/usr/bin/env python
"""
Created on Thu Oct 16 11:31:11 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from of.utils import Bunch
class InferenceAlgorithm(object):
    def __init__(self,ll_func,lp_func,use_ave_ll):
        self.ll_func=ll_func
        self.lp_func=lp_func
        self.record = Bunch()
        self.record.runs = []
        self.nRuns = 0
        self.use_ave_ll=use_ave_ll
    def get_record(self):
        return self.record        

if __name__ == "__main__":
    pass















