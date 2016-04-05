#!/usr/bin/env python
"""
Created on Thu Mar 13 17:13:45 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from _CpaSpace import my_dtype 
from of.utils import ObsoleteError
from of.utils import ipshell



class Multiscale(object):
    """
    You should never use this class directly.
    """
    def __init__(self,XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around=None,
                 zero_vals=None,
                 my_dtype=my_dtype,
                 Ngrids=None,
                 CpaCalcs=None,
                 only_local=False,
                 cont_constraints_are_separable=None):
        
        if zero_vals is None:
            raise ValueError("The child's __init__ must pass this argument")                       
        if Ngrids is None:
            raise ValueError("The child's __init__ must pass this argument")                 
        if CpaCalcs is None:
            raise ValueError("The child's __init__ must pass this argument")                     
        if warp_around is None:
            raise ValueError("You must pass this argument")
        if my_dtype not in [np.float32,np.float64]:
            raise ValueError(my_dtype)
                        
        try: # Test if iterable
            zero_v_across_bdry.__iter__
        except:
            raise  
        try: # Test if iterable
            base.__iter__
        except:
            raise               
        try: # Test if iterable
            warp_around.__iter__
        except:
            raise         
        
        if len(base)!=self.dim_domain:
            raise ValueError(base,self.dim_domain)
        if len(zero_v_across_bdry)!=self.dim_domain:
            raise ValueError(zero_v_across_bdry,self.dim_domain)
        if len(Ngrids)!=self.dim_domain:
            raise ValueError(Ngrids,self.dim_domain)
        if len(XMINS)!=self.dim_domain:
            raise ValueError(XMINS,self.dim_domain)
        if len(XMAXS)!=self.dim_domain:
            raise ValueError(XMAXS,self.dim_domain)
            
          

        self.XMINS = np.asarray(XMINS,dtype=my_dtype)
        self.XMAXS = np.asarray(XMAXS,dtype=my_dtype)                     
        if  (self.XMINS>=self.XMAXS).any():
            raise ValueError(XMINS,XMAXS)    

        if not np.allclose(self.XMINS,0):
            raise NotImplementedError(self.XMINS)  

       
 

        self.nLevels = nLevels        
        self.zero_v_across_bdry=zero_v_across_bdry       
        self.vol_preserve=vol_preserve 
        self.base=base
        
        self.L_cpa_space=[]            
        
        self.warp_around = warp_around            
        self.my_dtype=my_dtype        
        
        
        # self.calcs is shared by the cpa spaces in all levels
        self.calcs=CpaCalcs(XMINS=XMINS,XMAXS=XMAXS,
                            Ngrids=Ngrids,use_GPU_if_possible=True) 
        
        
        
        self.only_local = only_local
        
        if self.dim_domain>1:
            if cont_constraints_are_separable is None:
                raise ObsoleteError("""
                Expected True/False value for cont_constraints_are_separable;
                got None instead""")

    def try_to_determine_level(self):
        if self.nLevels==1:
            level=0
            return level
        else:
            msg="nLevels={}>1 but you didn't specify the level".format(self.nLevels)
            raise ValueError(msg)
    def __repr__(self):
        s = "Multiscale: (nLevels = {}\n".format(self.nLevels)
        s += repr(self.L_cpa_space)
        return s
#    def get_zeros_no_con(self):
#        return [sp.get_zeros_no_con() for sp in self.L_cpa_space]

    def get_zeros_PA(self,level):
        return self.L_cpa_space[level].get_zeros_PA()
    def get_zeros_PA_all_levels(self):
        return [sp.get_zeros_PA() for sp in self.L_cpa_space]


    def get_zeros_theta(self,level):
        return self.L_cpa_space[level].get_zeros_theta()
    def get_zeros_theta_all_levels(self):
        return [sp.get_zeros_theta() for sp in self.L_cpa_space]

    def update_pat(self,Avees,level):
        raise ObsoleteError("Use the update_pat_in_one_level method instead")
    def update_pat_in_one_level(self,Avees,level):
        self.L_cpa_space[level].update_pat(Avees=Avees)  
    def update_pats_in_all_levels(self,ms_Avees):        
        [cpa_space.update_pat(Avees=Avees) for 
         (cpa_space,Avees) in zip(self.L_cpa_space,ms_Avees)]         
    def update_pat_from_velTess_in_one_level(self,velTess,level):
        self.L_cpa_space[level].update_pat_from_velTess(velTess=velTess)
    def propogate_theta_coarse2fine(self,theta_coarse,theta_fine):
        """
        Modifies theta_fine        
        """
                                                                
       
#       
        dd = [sp.d for sp in self.L_cpa_space]
    
        level_coarse = dd.index(len(theta_coarse))
        level_fine = dd.index(len(theta_fine))
        
        if level_fine != level_coarse+1:
            raise ValueError(level_coarse,level_fine)
        
        sp_coarse = self.L_cpa_space[level_coarse]
        sp_fine = self.L_cpa_space[level_fine]

        Avees_coarse = sp_coarse.get_zeros_PA()
        Avees_fine = sp_fine.get_zeros_PA()
        
        sp_coarse.theta2Avees(theta_coarse,Avees_coarse)
        self.propogate_Avees_coarser2fine(Avees_coarse,Avees_fine)
        sp_fine.Avees2theta(Avees_fine,theta_fine)
       


    def propogate_Avees_coarser2fine(self,Avees_coarse,Avees_fine):        
        """
        Duplicates the values of Avees_coarse into Avess_fine.
        IMPORTANT: This modifies Avees_fine inplace.
        """       
        # DEBUG
#        Avees_coarse[:]=np.arange(Avees_coarse.size).reshape(Avees_coarse.shape) 
        
#        lengthAvee = self.L_cpa_space[0].lengthAvee    
        lengthAvee = self.lengthAvee                    
        # Remark: up.shape =  (number of As in the coarse level, lengthAvee  )    
              
        up=Avees_coarse.reshape(-1,lengthAvee) 
                        
        
#        N =  2**self.dim_domain
#        tmp = np.zeros((up.shape[0] *N,lengthAvee ))
        
        nAs = [sp.D / self.lengthAvee for sp in self.L_cpa_space]
        
        level_coarse = nAs.index(up.shape[0])
        level_fine = level_coarse+1
        
        sp_coarse = self.L_cpa_space[level_coarse]
        sp_fine = self.L_cpa_space[level_fine]
        
        if sp_coarse.tess != sp_fine.tess:
            raise ValueError(sp_coarse.tess , sp_fine.tess)
            
        if self.dim_domain == 1:
            coarse=Avees_coarse.reshape(sp_coarse.tessellation.nCx,-1)
            fine=Avees_fine.reshape(sp_fine.tessellation.nCx,-1)
            for i in range(sp_coarse.tessellation.nCx):              
                    fine[2*i]=coarse[i]
                    fine[2*i+1]=coarse[i]                   
                  
        elif self.dim_domain == 2:
            coarse=Avees_coarse.reshape(sp_coarse.tessellation.nCy,sp_coarse.tessellation.nCx,-1)
            fine=Avees_fine.reshape(sp_fine.tessellation.nCy,sp_fine.tessellation.nCx,-1)
            for i in range(sp_coarse.tessellation.nCy):
                for j in range(sp_coarse.tessellation.nCx):
                    # I know this works for rect. 
                    # I *think* it also works for tri.
                    # Todo: check that.
                    if sp_coarse.tess == 'II':                        
                        fine[2*i,j*2]=coarse[i,j]
                        fine[2*i,j*2+1]=coarse[i,j]
                        fine[2*i+1,j*2]=coarse[i,j]
                        fine[2*i+1,j*2+1]=coarse[i,j]
                    elif sp_coarse.tess == 'I':
                        # HBD: didn't check this. Should be fine...
                       
#                        from pylab import plt
#                        import pylab                        
#                        pylab.ion()
#                        plt.subplot(121);plt.imshow(sp_coarse.calc_tess());plt.colorbar()
#                        plt.subplot(122);plt.imshow(sp_fine.calc_tess());plt.colorbar()
                        cij = coarse[i,j]
                        ful=fine[2*i,j*2]
                        fur=fine[2*i,j*2+1]
                        fll=fine[2*i+1,j*2]
                        flr=fine[2*i+1,j*2+1]
                        
                        src=cij[:6]
                        np.copyto(dst=ful[:6],src=src)
                        np.copyto(dst=ful[6:6*2],src=src)
                        np.copyto(dst=fur[6*3:6*4],src=src)
                        np.copyto(dst=fur[6*0:6*1],src=src)
                        
                        src=cij[6:6*2]
                        np.copyto(dst=fur[6*1:6*2],src=src)
                        np.copyto(dst=fur[6*2:6*3],src=src)
                        np.copyto(dst=flr[6*0:6*1],src=src)
                        np.copyto(dst=flr[6*1:6*2],src=src)

                        src=cij[6*2:6*3]
                        np.copyto(dst=flr[6*2:6*3],src=src)
                        np.copyto(dst=flr[6*3:6*4],src=src)
                        np.copyto(dst=fll[6*1:6*2],src=src)
                        np.copyto(dst=fll[6*2:6*3],src=src)

                        src=cij[6*3:6*4]
                        np.copyto(dst=fll[6*3:6*4],src=src)
                        np.copyto(dst=fll[6*0:6*1],src=src)
                        np.copyto(dst=ful[6*2:6*3],src=src)
                        np.copyto(dst=ful[6*3:6*4],src=src)
                       
#                        ipshell('stop')
#                        1/0
                    else:
                        raise ValueError(sp_coarse.tess)
        elif self.dim_domain == 3: 
            if sp_coarse.tess=='I':              
                raise NotImplementedError("coarse to fine: tess='I', dim_domain=3")
                
            coarse=Avees_coarse.reshape(sp_coarse.tessellation.nCz,
                                        sp_coarse.tessellation.nCy,
                                        sp_coarse.tessellation.nCx,-1)
            fine=Avees_fine.reshape(sp_fine.tessellation.nCz,
                                    sp_fine.tessellation.nCy,
                                    sp_fine.tessellation.nCx,-1)
            for i in range(sp_coarse.tessellation.nCy):
                for j in range(sp_coarse.tessellation.nCx):
                    for k in range(sp_coarse.tessellation.nCz):
                        if sp_coarse.tess == 'II':                             
                            _coarse= coarse[i,j,k]                           
                            fine[2*i+0,2*j+0,2*k+0]=_coarse
                            fine[2*i+0,2*j+0,2*k+1]=_coarse
                            fine[2*i+0,2*j+1,2*k+0]=_coarse 
                            fine[2*i+0,2*j+1,2*k+1]=_coarse 
                            fine[2*i+1,2*j+0,2*k+0]=_coarse
                            fine[2*i+1,2*j+0,2*k+1]=_coarse
                            fine[2*i+1,2*j+1,2*k+0]=_coarse 
                            fine[2*i+1,2*j+1,2*k+1]=_coarse                             
                            
                        elif sp_coarse.tess == 'I':    
                            raise NotImplementedError("coarse to fine: tess='I', dim_domain=3")
            
                       
        else:
            raise NotImplementedError                    
                 
        # Note that since reshape doesn't copy, we already changed Avees_fine.    
       
 
   
       
