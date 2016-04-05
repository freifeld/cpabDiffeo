#!/usr/bin/env python
"""
Created on Tue Dec 16 09:16:36 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from of.utils import *
from of.gpu import GpuTimer

class TransformWrapper(object):
    """
    An abstract class
    """
    def __init__(self,
                 vol_preserve,
                 nLevels, 
                 base,
                 scale_spatial,
                 scale_value,
                 zero_v_across_bdry,
                 tess,
                 valid_outside,
                 only_local,
                 cont_constraints_are_separable=None):
        
        if not isinstance(vol_preserve,bool):
            raise TypeError(vol_preserve)
        if not isinstance(valid_outside,bool) and self.dim_domain>1:
            raise TypeError(valid_outside)
        

        
        self.nLevels=nLevels                     
        
        self.args = Bunch()
        self.args.vol_preserve = vol_preserve
        self.args.nLevels = nLevels 
        self.args.base = base
        self.args.scale_spatial = scale_spatial
        self.args.scale_value = scale_value
        self.args.zero_v_across_bdry = zero_v_across_bdry
        self.args.tess  =  tess
        self.args.valid_outside = valid_outside
        self.args.only_local = only_local
        if self.dim_domain>1:
            if cont_constraints_are_separable is None:
                raise ObsoleteError("""
                Expected True/False value for cont_constraints_are_separable;
                got None instead""")
            self.cont_constraints_are_separable=cont_constraints_are_separable
        
        self.timer = Bunch()
        
        self.timer.calc_T = Bunch()
        self.timer.calc_T.integrate_gpu = GpuTimer()
        self.timer.calc_T.expm_gpu = GpuTimer()
        
        self.timer.calc_T_simple = Bunch()
        self.timer.calc_T_simple.integrate_gpu = GpuTimer()
        
        
        self.timer.remap = GpuTimer()
        

        
    def __repr__(self):
        s = 'tw:\n\t'
        s += '\n\t'.join(repr(self.ms).splitlines())
        return s    
    
    def try_to_determine_level(self):
        return self.ms.try_to_determine_level()
    def get_zeros_theta(self,level):
        return self.ms.get_zeros_theta(level=level)
    def get_zeros_theta_all_levels(self):
        return self.ms.get_zeros_theta_all_levels()
    def get_zeros_PA(self,level):
        return self.ms.get_zeros_PA(level=level)
    def get_zeros_PA_all_levels(self):
        return self.ms.get_zeros_PA_all_levels()    
    def sample_from_the_ms_prior_coarse2fine_all_levels(self):
        """
        'ms' stands for multiscale
        """
        ms_Avees, ms_theta=self.msp.sampleCoarse2Fine() 
        return ms_Avees, ms_theta
    def sample_from_the_ms_prior_coarse2fine_one_level(self,ms_Avees, ms_theta,level_fine):
        if level_fine<=0:
            raise ValueError
        self.msp.sample_normal_in_one_level_using_the_coarser_as_mean(Avees_coarse=ms_Avees[level_fine-1],
                                                                                Avees_fine=ms_Avees[level_fine],
                                                                                theta_fine=ms_theta[level_fine],    
                                                                                level_fine=level_fine)

    def sample_gaussian(self, level,Avees, theta, mu):
        """
        Modifies Avees and theta
        """              
        self.msp.sample_normal_in_one_level(level, Avees, theta, mu)

    def sample_gaussian_velTess(self, level,Avees, velTess, mu):
        """
        Modifies Avees and velTess
        """              
        self.msp.sample_normal_in_one_level_velTess(level, Avees,velTess, mu)
        
    def update_pat_from_Avees(self,Avees=None,level=None):
        if level is None:
            level = self.try_to_determine_level()                                 
        self.ms.update_pat_in_one_level(Avees=Avees,level=level)            
                   
    def update_ms_pats(self,ms_Avees=None):
        if ms_Avees is None:
            raise NotImplementedError
        self.ms.update_pats_in_all_levels(ms_Avees=ms_Avees)

    def update_pat_from_velTess(self,velTess,level=None):
        if level is None:
            raise ValueError("I am not going to try guess what level you want")
        self.ms.update_pat_from_velTess_in_one_level(velTess=velTess,
                                                     level=level)   

    def calc_cell_idx(self,pts_src,cell_idx,level,permute_for_disp=False):
        cpa_space=self.ms.L_cpa_space[level]
        cpa_space.calc_cell_idx(pts_src,cell_idx)
        if permute_for_disp:
            cell_idx.gpu2cpu()
            
            p=cpa_space.tessellation.permuted_indices
            cell_idx2=np.zeros_like(cell_idx.cpu)
            for c in range(cpa_space.nC):             
                cell_idx2[cell_idx.cpu==c]=p[c]              
            np.copyto(dst=cell_idx.cpu,src=cell_idx2)            
            cell_idx.cpu2gpu()
            



    def calc_v(self,level,pts=None,v=None):
         
        ms=self.ms
        if pts is None and v is None:  
#            ipshell(__file__)
            if self.dim_domain == 1:
                pts,v = self.x_dense,self.v_dense
            else:
                pts,v = self.pts_src_dense,self.v_dense
            
        elif pts is not None and v is not None:
            if pts.shape != v.shape:
                raise ValueError(pts.shape,v.shape)
            
        else:
            raise ValueError("pts and v must be either both None or both not None")            
        cpa_space = ms.L_cpa_space[level]       
        cpa_space.calc_v(pts=pts,out=v )

    def calc_T(self,*args,**kwargs):
        raise ObsoleteError("Use calc_T_inv or calc_T_fwd instead")
        
    def calc_T_fwd_simple(self,pts_src,pts_fwd,level,timeit=False,
                  int_quality=0):
        """
        pts_src and pts_fwd are CpuGpuArrays.        
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """                      
        self._calc_T_simple(pts_src,pts_fwd,mysign=1,level=level,
                    timeit=timeit,int_quality=int_quality)        
        
    def calc_T_fwd(self,pts_src,pts_fwd,level,timeit=False,
                  int_quality=0):
        """
        pts_src and pts_fwd are CpuGpuArrays.        
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """                          
        self._calc_T(pts_src,pts_fwd,mysign=1,level=level,
                    timeit=timeit,int_quality=int_quality)
    def calc_T_inv(self,pts_src,pts_inv,level,timeit=False,
                  int_quality=0):
        """
        pts_src and pts_inv are CpuGpuArrays.        
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """                      
        self._calc_T(pts_src,pts_inv,mysign=-1,level=level,
                    timeit=timeit,int_quality=int_quality)

    def calc_T_inv_simple(self,pts_src,pts_fwd,level,timeit=False,
                  int_quality=0):
        """
        pts_src and pts_fwd are CpuGpuArrays.        
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """                      
        self._calc_T_simple(pts_src,pts_fwd,mysign=-1,level=level,
                   timeit=timeit,int_quality=int_quality)        
        

    def _calc_T(self,pts_src,pts_transformed,mysign,level,
              timeit=False,int_quality=0):

        """
        pts_src and pts_transformed are CpuGpuArrays.
        mysign: -1/+1.   
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """
         
        ms=self.ms
        
        if int_quality not in [-1,0,1]:
            raise ValueError
        params_flow_int = [self.params_flow_int,
                           self.params_flow_int_fine,
                           self.params_flow_int_coarse][int_quality]
                           
        
#        print params_flow_int                   
#        if int_quality==-1:
#            params_flow_int = self.params_flow_int_coarse
#        elif int_quality==0:
#            params_flow_int = self.params_flow_int
#        elif int_quality==1:
#            params_flow_int = self.params_flow_int_fine
#        else:
#            raise ValueError
#        if ms.nLevels>1:
#            raise NotImplementedError("I will fix it later. For now use one level")
        cpa_space = ms.L_cpa_space[level]
#        if my_timer:
#            print 'Level',level
         
#            print 'level: ',level
#        if my_timer:
#            tic_T=time.clock()
        
        if timeit:
            timer=self.timer.calc_T
        else:
            timer=None
        cpa_space._calc_T(pts = pts_src,mysign=mysign,
                          out=pts_transformed,
                          timer=timer,
                          **params_flow_int)                                                                 
                                              
        
#        if my_timer:
#            toc_T=time.clock()
#            print "Time (transform):",toc_T-tic_T


    def _calc_T_simple(self,pts_src,pts_transformed,mysign,level,
               timeit=False,int_quality=0):
        """
        pts_src and pts_transformed are CpuGpuArrays.
        mysign: -1/+1.   
        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
        my_timer: True/False   
        int_quality is in [-1,0,1]        
        """
         
        ms=self.ms
        
        if int_quality not in [-1,0,1]:
            raise ValueError
        params_flow_int = [self.params_flow_int,
                           self.params_flow_int_fine,
                           self.params_flow_int_coarse][int_quality]

        cpa_space = ms.L_cpa_space[level]

        if timeit:
            timer=self.timer.calc_T_simple
        else:
            timer=None
        cpa_space.calc_T_simple(pts = pts_src,mysign=mysign,
                         out=pts_transformed,
                         timer=timer,**params_flow_int)                                                                 
                                             
        
          
            
            




if __name__ == "__main__":
    pass
