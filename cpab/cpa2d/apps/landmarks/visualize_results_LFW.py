#!/usr/bin/env python
"""
Created on Thu Mar 24 09:45:20 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import pylab
from pylab import plt
from of.utils import *
from of.gpu import CpuGpuArray
import of.plt
from cpab.cpa2d.TransformWrapper import TransformWrapper

from get_data_LFW import get_data
from disp import disp

if not inside_spyder():
    pylab.ion()

name = 'LFW_5_to_6'
data = get_data(name)
src = CpuGpuArray(data.src)
dst = CpuGpuArray(data.dst)
transformed = CpuGpuArray.zeros_like(src)

fname_results = os.path.splitext(data.fname)[0]+'_result.pkl'
FilesDirs.raise_if_dir_does_not_exist(os.path.dirname(fname_results))
print 'Loading',fname_results
results=Pkl.load(fname_results)
theta_est = results.theta


tw = TransformWrapper(**results.tw_args)
tw.create_grid_lines(step=0.1,factor=0.5)
scale_quiver=1000 # The *smaller* this value is, the larger the plotted arrows will be.

level=-1 # pick the finest scale

cpa_space = tw.ms.L_cpa_space[level]
cpa_space.theta2Avees(theta_est)
cpa_space.update_pat() 
tw.calc_T_fwd(src,transformed,level=level)
transformed.gpu2cpu()
tw.calc_v(level=level)
tw.v_dense.gpu2cpu()

plt.close('all')
disp(tw=tw,theta=theta_est,src=src,dst=dst,transformed=transformed,level=level,
     use_subplots=1,scale_quiver=scale_quiver)
     
     
if not inside_spyder():       
    raw_input("Press Enter to exit")    
        