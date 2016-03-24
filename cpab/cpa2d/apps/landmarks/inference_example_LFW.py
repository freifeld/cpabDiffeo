#!/usr/bin/env python
"""
Created on Wed Mar 23 10:28:32 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
import pylab

from pylab import plt
plt.close('all')
from of.gpu import CpuGpuArray
from of.utils import *
import of.plt

from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter

from get_data_LFW import get_data
from InferenceParamsLFW import InferenceParams

name = 'LFW_5_to_6'
data = get_data(name)

if not inside_spyder():
    pylab.ion()

class Options(object):
    axis_ij=False
    gt=None
    def __init__(self,name):
        name = name.lower()
        self.axis_ij=True        
        self.nCols=256 # TODO: find out what was the acutal size
        self.nRows=256 # TODO: find out what was the acutal size
    
options = Options(name)    
    
data.src = CpuGpuArray(data.src)
data.dst = CpuGpuArray(data.dst)        
    
inference_params = InferenceParams()

# quick debug
#inference_params.MCMCniters_per_level=10

tf = TransformationFitter(nRows=options.nRows,
                          nCols=options.nCols,
                          vol_preserve=inference_params.vol_preserve,
                          sigma_lm =inference_params.sigma_lm,
                          base=inference_params.base,
                          nLevels=inference_params.nLevels,
                          valid_outside=inference_params.valid_outside,
                          tess=inference_params.tess,
                          scale_spatial=inference_params.scale_spatial,
                          scale_value=inference_params.scale_value,   
                          scale_quiver=inference_params.scale_quiver,
                          zero_v_across_bdry=inference_params.zero_v_across_bdry)

tw = tf.tw



tf.set_dense()       


          
tf.set_data(data)   
tf.set_run_lengths([inference_params.MCMCniters_per_level]*tf.nLevels)            

theta,inference_record = tf.fit(use_prior=inference_params.use_prior,
                                proposal_scale=inference_params.proposal_scale,
                                use_local=inference_params.use_local)                   


theta_est=theta.copy()


fname_results = os.path.splitext(data.fname)[0]+'_result.pkl'
FilesDirs.raise_if_dir_does_not_exist(os.path.dirname(fname_results))
tosave = {'tw_args':inference_record.tw_args,
              'theta':inference_record.theta}
tosave = Bunch(**tosave)
Pkl.dump(fname_results,tosave,override=True)




if not inside_spyder():       
    raw_input("Press Enter to exit")    
    
    
