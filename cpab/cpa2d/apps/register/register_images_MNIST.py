#!/usr/bin/env python
"""
Created on Fri Dec  5 10:41:33 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pylab import plt
from cpab.cpa2d.inference.transformation.Register import Register
from of.utils import *
import pylab
from get_data import get_data

if not inside_spyder():
    pylab.ion()
 

def main(data,infernece_params,dispOn ):
 
    img1=data.img1 # src
    img2=data.img2 # dst
    
    if img1.shape != img2.shape:
        raise ValueError(img1.shape,img2.shape)

    nRows,nCols = img1.shape[:2]  
    nPts = nRows * nCols 
      
    reg = Register(nRows=nRows,
                   nCols=nCols,
                   base=inference_params.base,
                   nLevels=inference_params.nLevels, tess='I',
                   zero_v_across_bdry=[False,False],
                   sigma_signal=inference_params.sigma_signal ,                   
                   scale_spatial=inference_params.scale_spatial,
                   scale_value=inference_params.scale_value,
                   wlp=inference_params.wlp,                   
                   ll_type=inference_params.ll_type,
                   only_local=False,
                   valid_outside=infernece_params.valid_outside)  
                 
    reg.set_dense(domain_start=0,domain_end=nPts)
                        
    reg.set_data(x=reg.tw.pts_src_dense,
                 signal_src=img1,signal_dst=img2,
                 isbinary=inference_params.isbinary)
    
     
    print reg
    
    
#    reg.set_run_lengths([50000]*reg.nLevels)
    reg.set_run_lengths([infernece_params.MCMCniters_per_level]*reg.nLevels)
        
#    if inside_spyder():
#        reg.set_run_lengths([100]*reg.nLevels)
#    #        reg.set_run_lengths([50000]*reg.nLevels)
#    else:        
#        reg.set_run_lengths(run_lengths)
    
    theta_est,inference_record = reg.fit(use_prior=infernece_params.use_prior,
                                     use_local=infernece_params.use_local,
                                     dispOn=dispOn,
                                     #interp_type_for_ll=cv2.INTER_LANCZOS4,
                                     interp_type_for_ll= 'gpu_linear',
                                     interp_type_during_visualization=cv2.INTER_LANCZOS4)
     
    if dispOn:
        plt.figure(1000)
        plt.clf()
        reg.plot_inference_summary(inference_record)    
    
    
    reg.transformed.gpu2cpu()
    reg.signal.transformed.gpu2cpu()
    
   
    inference_record.transformed = reg.transformed.cpu
    inference_record.src = reg.src.cpu
    inference_record.signal = Bunch()
    inference_record.signal.src = reg.signal.src.cpu
    inference_record.signal.dst = reg.signal.dst.cpu
    inference_record.signal.transformed = reg.signal.transformed.cpu
    

    
#    if TF.dump_results:
#    Pkl.dump(results_filename,inference_record,create_dir_if_needed=1,override=1)

     
    return  reg,inference_record,theta_est
    
if __name__ == "__main__":
    from pyvision.essentials import *

    class InferenceParamsDefault(object):
        MCMCniters_per_level = 10000
        isbinary = False   

        valid_outside=True
        # values for the prior
        scale_spatial = 10  # the larger the smoother
        scale_value = 500 # variance
        
        wlp=1e-3 # weight of log prior
        use_prior=True            
        use_local = True
        base=[2,2]
        nLevels=3        
        imresize_factor = 2 # >1 is upsamplling. 
                                 # <1 is downsampling        
#     
#    class InferenceParams(InferenceParamsDefault):
#        def __init__(self):
##            self.base=[2,2];self.nLevels=2
##            self.base=[3,3];self.nLevels=3
###            self.base=[9,9];self.nLevels=1;self.MCMCniters_per_level=30000
##            self.base=[9,9];self.nLevels=1;self.MCMCniters_per_level=50000
#            self.base=[8,8];self.nLevels=1;self.MCMCniters_per_level=20000
##            self.base=[4,4];self.nLevels=2;self.MCMCniters_per_level=10000
##            self.base=[2,2];self.nLevels=1;self.MCMCniters_per_level=1000
#
##            self.base=[1,1]
#
#            self.imresize_factor=1.5
##            self.imresize_factor=2
#            
#            self.isbinary=False
#            if self.isbinary:
#                self.sigma_signal = 1 * self.imresize_factor
#            else:
#                self.sigma_signal = 50
#            self.ll_type=['gaussian','gaussian_on_distancetransform'][self.isbinary]
#            
        
    class InferenceParams(InferenceParamsDefault):
        def __init__(self):
            self.base = [4,4]; self.nLevels = 1; self.MCMCniters_per_level = 10000

            self.imresize_factor = 1.5
            
            self.isbinary=False
            if self.isbinary:
                self.sigma_signal = 1 * self.imresize_factor
            else:
                self.sigma_signal = 50
            self.ll_type=['gaussian','gaussian_on_distancetransform'][self.isbinary]
                    
        
    inference_params = InferenceParams()  
     
    name='MNIST_one_00001_to_00015'
#    name =  'MNIST_four_00009_to_00013'
    name =  'MNIST_four_00013_to_00009'
#    name =  'MNIST_four_00022_to_00009'
    data = get_data(name=name,
                    imresize_factor=inference_params.imresize_factor)

    dname_results = os.path.join(HOME,'data/derived/MNIST/examples',name)
    FilesDirs.mkdirs_if_needed(dname_results)
#    fname_results = os.path.join(dname_results ,get_time_stamp()+'.pkl')
    
    fname_results = os.path.join(dname_results ,'result.pkl')


    dispOn = False or 1
    reg,inference_record,theta_est = main(data,inference_params,dispOn=dispOn)


    


    tosave = {'tw_args':inference_record.tw_args,
              'theta':inference_record.theta,
              'signal':inference_record.signal,
              'imresize_factor':inference_params.imresize_factor}
    tosave = Bunch(**tosave)
    
    
    Pkl.dump(fname_results,tosave,override=True)

    
    if not inside_spyder():       
        raw_input("Press Enter to exit")    
    
