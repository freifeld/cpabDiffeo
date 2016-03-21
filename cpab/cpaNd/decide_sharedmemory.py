#!/usr/bin/env python
"""
Created on Thu Mar 10 13:07:45 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from of.utils import computer

def decide_sharedmemory(dim_domain,dim_range,nC):
    """
    Output is in [0,1,2].
    0: won't used shared memory.
    1: will use shared memory just for the A's. 
    2: will use shared memory for both the A's and the expm's. 
    Besides the arguments to this functions (i.e.: dim_domain;dim_range;nC),
    the answer should also depend on your graphics card. 
    Trying to infer this automatically turned out to be harder than expected.
    I thus suggest you will tweak this script to match you card.
    The safest choice is to just pick 0, but then you may lose some speed.
    """
    
    if dim_domain==1:
        if computer.has_good_gpu_card==0:
            sharedmemory=2
        else:            
            raise NotImplementedError(computer.has_good_gpu_card, dim_domain,  nC)
    elif dim_domain==2:
        if computer.has_good_gpu_card==0 and  nC <= 4**4:
            sharedmemory=2
        elif computer.has_good_gpu_card==0 and  nC <= 4**5:
            sharedmemory=1
        elif computer.has_good_gpu_card==0 and dim_domain ==2:
            sharedmemory=0
    
        elif computer.has_good_gpu_card and  nC <= 4**5:
            sharedmemory=2
        else:
            raise NotImplementedError(computer.has_good_gpu_card, dim_domain,  nC)
    elif dim_domain==3:
        if computer.has_good_gpu_card==0 and  nC <= 4**4: 
            sharedmemory=2  
        elif computer.has_good_gpu_card==0 and  nC <= 4**5: 
            sharedmemory=1  
        else:
            raise NotImplementedError(computer.has_good_gpu_card, dim_domain,  nC)
    else:
        raise NotImplementedError(computer.has_good_gpu_card, dim_domain,  nC)
    return sharedmemory        
            



if __name__ == "__main__":
    for nC in [4**x for x in range(6)]:
        sharedmemory=decide_sharedmemory(dim_domain=2,dim_range=2,nC=nC)


        print nC, sharedmemory