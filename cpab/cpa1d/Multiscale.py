#!/usr/bin/env python
"""
Created on Mon Feb  3 18:27:15 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from cpab.cpa1d.CpaSpace import CpaSpace
from cpab.cpaNd import Multiscale as MultiscaleNd
from of.gpu import CpuGpuArray
from of.utils import ipshell

from cpab.cpa1d.calcs import CpaCalcs 
 
class Multiscale(MultiscaleNd):
    dim_domain=1
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo       
    def __init__(self,XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels=3,
                 base=[2],
                 warp_around=[False],my_dtype=np.float64
#                ,calcs=None
                 ,Ngrids=None
#                ,CpaCalcs=None
                 ,**kwargs):                       
        if 'CpaCalcs' in kwargs.keys():
            raise ObsoleteError
                    
        if nLevels == 1 and base == 1 and zero_v_across_bdry[0]:
                raise ValueError("This won't fly") 

        
        super(Multiscale,self).__init__(XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around,
                 my_dtype,
                 Ngrids=Ngrids,
                 CpaCalcs=CpaCalcs) 
                 
        base = np.asarray(base)    
        
        for i in range(nLevels):    
##            if i==0:
##                continue
##            if not zero_vx_across_bdry :
##                n=base*2**(i)
##            else:
##                n=base*2**(i+1)
#            n = base * (2**i)
#            nCx=n  
##            print i,':',nCx,nCy     
##            print zero_v_across_bdry  
#
#
#            cpa_space = CpaSpace(XMINS,XMAXS,[nCx], 
#                                zero_v_across_bdry, 
#                                vol_preserve,warp_around,my_dtype,calcs=calcs) 
        
#            if not all(zero_v_across_bdry):
#                nCellsInEachDim=base*2**(i)
#            else:
#                nCellsInEachDim=base*2**(i+1)
              
            nCellsInEachDim=base*2**(i)
#            print nCellsInEachDim,base,2**(i),nLevels
#            1/0
            cpa_space = CpaSpace(XMINS,XMAXS,nCellsInEachDim,
                                zero_v_across_bdry,
                                vol_preserve,
                                warp_around=warp_around,cpa_calcs=self.calcs)         
            if cpa_space.d <= 0:
                raise ValueError            
            self.L_cpa_space.append(cpa_space)
            
        self.Aveess = [cpa_space.Avees for cpa_space in self.L_cpa_space] 
        self.pats = [cpa_space.pat for cpa_space in self.L_cpa_space] 
        self.Ass = [cpa_space.As for cpa_space in self.L_cpa_space] 
        
#    def __repr__(self):
#        s = "Multiscale cpav info:"
#        s+= '\nx_{}_{}_y_{}_{}'.format(self.xmin,self.xmax,self.ymin,self.ymax)
#        s += '\n\tzero_vx_across_bdry imposed: {}'.format([False,True][self.zero_vx_across_bdry])
#        s += '\n\tzero_vy_across_bdry imposed: {}'.format([False,True][self.zero_vy_across_bdry])
#        s += '\n\tvolume-preserving imposed: {}'.format([False,True][self.vol_preserve])
#               
#        s+='\nnLevels={}'.format(self.nLevels)
#        for cpav in self.L_cpa_space:
#            s+='\n'+'\n'.join(repr(cpav).splitlines()[:3])
#
#        return s
          
 

    def get_pts_evenly_spaced(self,nPts):
        """
        TODO: it seems I have some bug with the endpoints.
        So for now I took it out 
        """
        raise ObsoleteError("Use get_x_dense instead")      
        
    def get_x_dense(self,nPts):
        return self.L_cpa_space[0].get_x_dense(nPts)
 

from pyvision.essentials import *    
#def main(): 
if __name__ == '__main__':
    
     
    from cpab.prob_and_stats.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
    from cpab.cpa1d.calcs import * 
    
#    os.system('touch ~/{0}.tmp'.format(computer.hostname))
    import pylab   
    from pylab import plt
    
         
    if  computer.has_good_gpu_card != 0:
        pylab.ion()

    np.random.seed(15)
    
    cv2destroyAllWindows()
    plt.close('all')  
    
    XMINS = [0]
    XMAXS = [1] 
    
    
    class Conf0:
        zero_v_across_bdry=[False]           
        vol_preserve = False
        warp_around=[True]
    class Conf1:
        zero_v_across_bdry=[True]           
        vol_preserve = False 
        warp_around=[False]

    Conf = [Conf0,Conf1][1]
 

    Ngrids=[1000]
      
    ms=Multiscale(XMINS,XMAXS,Conf.zero_v_across_bdry, 
                  Conf.vol_preserve,nLevels=2,base=[100],
                  warp_around=Conf.warp_around,
                  Ngrids=Ngrids
#                  ,CpaCalcs=CpaCalcs 
                  )
#                              
    
     
    class TF0:
        do_grid=False 
        plot_cell_bdry=False
       
        plot_ini_pts=False
        
#    
    class TF1:
        do_grid=True
        plot_cell_bdry=False
       
        plot_ini_pts=False
         

    class TF([TF0,TF1][1]):
        savefigs=False
        
    
    params_flow_int = get_params_flow_int()
#    params_flow_int.dt = 0.01
#    params_flow_int.nTimeSteps = 1.0 / params_flow_int.dt   
#    params_flow_int.nStepsODEsolver=  100    
    print params_flow_int
    
#    return ms
    msp=MultiscaleCoarse2FinePrior(ms,scale_spatial=1.0 * 10,
                                       scale_value=2,                          
                                       left_blk_std_dev=1,right_vec_scale=1)
                                       
#    sample_Avees_all_levels, sample_cpav_all_levels=msp.sampleCoarse2Fine()            
    
    samps = [msp.sampleCoarse2Fine()    ,
             msp.sampleCoarse2Fine()    ]                                                           
 
   
   
    a,b = samps
    vals = []
    for i,weight in enumerate([0.5]):
#    for i,weight in enumerate(np.linspace(0,1,21)):
        
        vals.append( copy.deepcopy(a))
        for level in range(ms.nLevels):
     
            vals[i][0][level] = weight*a[0][level]+(1-weight)*b[0][level]
            vals[i][1][level] = weight*a[1][level]+(1-weight)*b[1][level]
     
 
    
    for i_s,(sample_Avees_all_levels, sample_cpav_all_levels) in enumerate(vals):
 
        
        
        print "Transform pts"
        for level,cpa_space in enumerate(ms.L_cpa_space):
            print 'level: ',level
            print cpa_space
            
            # in general, this doesn't have to evenly spaced. Just in the right range.
            x_dense=cpa_space.get_x_dense(nPts=1000)
 
          
            # This needs to be evenly spaced. 
            interval = np.linspace(-3,3,x_dense.size)   


#            # There is actually a reason why I do the funny thing below    
##            x_select = x[::100/2].copy()
#            x_select = x.copy()[::-1][::100/2][::-1]
#            
#            
#            # but now that I need to copy, it may not be relevant any more...
#            x_select = x_select.copy()
            
#            x_select = x[::100/2].copy()
            x_select = CpuGpuArray(x_dense.cpu[::100/2].copy())
            
    #        x_select[:]=np.linspace(.1,1.0,len(x_select))
    #        x_select[3:-1] +=.1
    #        x_select[6:-1] -=.05 
    
            
                                                    
#            pat = PAT(pa_space=cpa_space,Avees=sample_Avees_all_levels[level]) 
            cpa_space.update_pat(Avees=sample_Avees_all_levels[level])
#           
            
            v_dense = CpuGpuArray.zeros_like(x_dense)
            cpa_space.calc_v(pts=x_dense,out=v_dense)
            
               

            src=x_dense
            
            print '#pts =',len(src)
            
            transformed = CpuGpuArray.zeros_like(src)
            tic=time.clock() 
            cpa_space.calc_T(pts = src, mysign=1,out=transformed,**params_flow_int)             
            toc = time.clock()
            print "time (src)",toc-tic             

                     
            src_select =  x_select
            transformed_select = CpuGpuArray.zeros_like(src_select)
            
            print '#pts =',len(src_select)
            tic=time.clock()   
            cpa_space.calc_T(pts = src_select, mysign=1,out=transformed_select,**params_flow_int)                                                 
            toc = time.clock()
            print "time (src select)",toc-tic    
            
            
#            src,transformed = hx[0,:,0],hx[-1,:,0]
#            
#            
#            src_select,transformed_select = hx_select[0,:,0],hx_select[-1,:,0]
             
            
            # for display
            transformed_select.gpu2cpu()
            transformed.gpu2cpu()
            v_dense.gpu2cpu()
           
            if 0:
                plt.figure(17)
                for c,A in enumerate(As):        
                    _x = np.ones((2,100))
                    m=cpa_space.cells_verts[c,0,0]
                    M=cpa_space.cells_verts[c,1,0]
                    _x[0] = np.linspace(m,M,100)
                    _v = A.dot(_x).flatten()
                    plt.plot(_x[0],_v) 
                      
            if 1:
#                plt.figure()
                plt.figure(1);plt.clf()
                plt.subplot(231)                 
                plt.plot(interval,src.cpu)   
                plt.title('src')
                plt.subplot(232)         
#                plt.plot(interval[1:],np.diff(src)/(interval[1]-interval[0]))
                dx=interval[1]-interval[0]
                plt.plot(interval[1:],np.diff(src.cpu.ravel())/dx)
                plt.title(" d/dx src")  
                plt.ylim(0,.5)
                plt.subplot(233)
                plt.plot(np.linspace(cpa_space.XMINS[0],cpa_space.XMAXS[0],
                                     interval.size),v_dense.cpu.ravel())
                plt.ylim(-1,1)
                plt.title('velocity')           
                plt.subplot(234)
                plt.plot(interval,transformed.cpu)
                plt.ylim(0,1)
                plt.title('transformed')  
                plt.subplot(235)           
#                plt.plot(interval[1:],np.diff(transformed)/(interval[1]-interval[0]))
                dx=interval[1]-interval[0]
                plt.plot(interval[1:],np.diff(transformed.cpu.ravel())/dx)                
                plt.ylim(0,5)
                plt.title('d/dx transformed') 
                
    
            
     
            else: 
          
    
                nPtsDense = x_dense.size
        
                plt.figure() 
                plt.subplot(231)      
                ind = 1+np.arange(len(x_select))  # the x locations for the groups               
                width = 0.4       # the width of the bars                
                ax = plt.gca()
                rects1 = ax.bar(ind,src_select.cpu, width, color='r')        
                   
        #        plt.plot(interval,src)   
                plt.title('src')                                    
                plt.subplot(232)    
                      
                       
                ax = plt.gca()
                
                 
                rects1 = ax.bar(ind,np.hstack([src_select[0],
                                               np.diff(src_select[:,0])]), width, color='r')
                plt.title(" diff src")
                plt.ylim(0,.5)
                
                plt.subplot(233)
#                plt.plot(np.linspace(cpa_space.XMINS[0],cpa_space.XMAXS[0],x.size),v.flatten())
                plt.plot(interval,v_dense.flatten())
                plt.title('velocity')     
                plt.subplot(234)      
                          
                
                ax = plt.gca()
                rects1 = ax.bar(ind,transformed_select, width, color='r') 
                plt.title('transformed')                   
                plt.subplot(235)                                                   
                ax = plt.gca()
                
                transformed_select_diff = np.hstack([transformed_select[0],np.diff(transformed_select[:,0])])

                rects1 = ax.bar(ind,transformed_select_diff, width, color='r')
                plt.ylim(0,.5)
                plt.title('diff transformed')         
            
            dirname_fig = os.path.join(HOME,'Dropbox','tmp')
            FilesDirs.mkdirs_if_needed(dirname_fig)
            filename_fig = os.path.join(dirname_fig,'{0:03}.jpg'.format(i_s))
#            print "sleep(2)"
#            time.sleep(10)
#            print "wake up"
            print "saving" ,filename_fig
            

            plt.savefig(filename_fig)
           
      
#if __name__ == "__main__":  
#    if 0 and computer.hostname in ['biscotti']:
#        f = os.path.join(os.path.abspath(os.path.dirname(__file__)),__file__)
#        cmd = "ssh pastelito 'python {0}'".format(f)
#        print cmd
#        os.system(cmd)
#    else:
#        ret = main()
#    
#    if 0 and not (computer.hostname in ['biscotti']):
#        raw_input("Press Enter to finish.")
             
