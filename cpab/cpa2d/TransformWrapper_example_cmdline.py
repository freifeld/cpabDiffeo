#!/usr/bin/env python
"""
Created on Mon Mar 14 16:13:10 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import argparse   
parser = argparse.ArgumentParser()
parser.add_argument('-i','--img',
                        nargs='?',
                        const=None,default=None)
parser.add_argument('-t','--tess',
                        nargs='?',
                        const='I',default='I')
d = {'eci':'eval_cell_idx','ev':'eval_v','showptsds':'show_downsampled_pts'}
for k,v in d.iteritems():
    parser.add_argument('-'+k,'--'+v,nargs='?',type=int,
                        const=1,default=1)  
    
parser.add_argument('-vo','--valid_outside',
                        nargs='?',type=int,
                        const=1,default=1)  
                        
parser.add_argument('-b','--base',type=int,
                        nargs=2,default=(1,1)) 

parser.add_argument('-vp','--vol_preserve',
                        nargs='?',type=int,
                        const=0,default=0)  
                        
parser.add_argument('-zbdry','--zero_v_across_bdry',type=int,
                        nargs=2,default=(0,0)) 

parser.add_argument('-ulwp','--use_lims_when_plotting',
                        nargs='?',type=int,
                        const=1,default=1)  


parser.add_argument('-nl','--nLevels',
                        nargs='?',type=int,
                        const=3,default=3)     

args = parser.parse_args()  
#for k,v in args.__dict__.iteritems():
#    print k,':',v , type(v)

# It is better to do this import here than in the in the begining.
# Otherwise, even calling the -h option can be very slow.
from TransformWrapper_example import example
from of.utils import inside_spyder
tw=example(**args.__dict__)
if not inside_spyder():
    raw_input('Press Enter to exit')
