#!/usr/bin/env python
"""
Created on Wed Mar 16 15:04:22 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


class Tessellation(object):
    def create_verts_and_H(self,dim_range,
              valid_outside
                              ):  
        """      
        H encodes the n'bors info.
        """    
        if self.type == 'I':
            return self.create_verts_and_H_type_I(dim_range,valid_outside)
        elif self.type=='II':
            return self.create_verts_and_H_type_II(dim_range)
        else:
            raise NotImplementedError(self.type)
    @staticmethod
    def make_it_hashable(arr):
        return tuple([tuple(r.tolist()) for r in arr]) 

if __name__ == "__main__":
    pass
