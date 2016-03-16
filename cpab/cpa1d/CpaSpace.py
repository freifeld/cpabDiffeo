#!/usr/bin/env python
"""
Created on Sun Mar  9 21:43:46 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
#from cpab.essentials import *
from scipy import sparse
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpaNd import CpaSpace as CpaSpaceNd

from cpab.cpaNd.utils import null
from cpab.cpa1d.utils import *     
     
my_dtype = np.float64
  
 
from cpab.cpa1d.Visualize import Visualize
 
from cpab.cpa1d.Tessellation import Tessellation
 
class CpaSpace(CpaSpaceNd):
    dim_domain=1
    dim_range=1
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo
                    
    def __init__(self,XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve,warp_around=None,my_dtype=my_dtype,
                 cpa_calcs=None):

#        XMINS = [-x for x in XMAXS] # DEBUG
        
        if  vol_preserve:
            raise ValueError("Doesn't make sense to use it for the 1D case")
        self.warp_around=warp_around            
        super(CpaSpace,self).__init__(XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve=vol_preserve,
                 warp_around=warp_around,
                 zero_vals=[],
                 my_dtype=my_dtype,
                 cpa_calcs=cpa_calcs)                      
         
        nCx=int(nCs[0])
       
        nC = nCx  # of cells
#        self.nC=nC 
#        self.nCx=nCx
#        
        tessellation = Tessellation(nCx=nCx,XMINS=self.XMINS,XMAXS=self.XMAXS)
        self.tessellation=tessellation
        
#        cells_multiidx,cells_verts=create_cells(nCx,self.XMINS,self.XMAXS) 
 
        
        try:
#            raise FileDoesNotExistError('fake error')
            subspace=Pkl.load(self.filename_subspace,verbose=1)
            B=subspace['B']
 
            nConstraints=subspace['nConstraints']
            nEdges=subspace['nEdges']
            constraintMat=subspace['constraintMat']
#            cells_verts =np.asarray(cells_verts)
            
        except FileDoesNotExistError: 
            print "Failed loading, so compute from scrartch"
             
            verts1,H,nEdges,nConstraints = self.tessellation.create_verts_and_H()
#            verts1,H,nEdges,nConstraints = create_verts_and_H(nC,cells_multiidx,cells_verts,
#                                                         dim_domain=self.dim_domain) 
        
#            raise ValueError(tessellation.cells_verts_homo_coo)
#            cells_verts=np.asarray(cells_verts) 
                                 
            L = create_cont_constraint_mat(H,verts1,nEdges,nConstraints,nC,dim_domain=self.dim_domain)                  
            
            if zero_v_across_bdry[0]:
                Lbdry = self.tessellation.create_constraint_mat_bdry(
#                                               XMINS,XMAXS,
#                                                   cells_verts, nC,
#                                              dim_domain=self.dim_domain,
                                      zero_v_across_bdry=self.zero_v_across_bdry)
                L = np.vstack([L,Lbdry])
                nConstraints += Lbdry.shape[0]
            if self.warp_around[0]: 
                Lwa = create_constraint_mat_warp_around(cells_verts,
                                                          nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lwa])
                nConstraints += Lwa.shape[0]
                 
            
            if vol_preserve:
                Lvol = create_constraint_mat_preserve_vol(nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lvol])
                nConstraints += Lvol.shape[0]
            try:
                B=null(L)     
            except:
                print '-'*30
                print self.filename_subspace
                print '-'*30
                raise
            constraintMat=sparse.csr_matrix(L)
            Pkl.dump(self.filename_subspace,{'B':B,
                                             'nConstraints':nConstraints,
                                             'nEdges':nEdges,
                                             'constraintMat':constraintMat},
                                             override=True)
 
        super(CpaSpace,self).__finish_init__(
                     tessellation=tessellation,
                        constraintMat=constraintMat,
                        nConstraints=nConstraints,
                        nIterfaces=nEdges,B=B,zero_vals=[])        
        if self.local_stuff is None:
            raise ValueError("WTF?")


        # MOVED THIS TO TESSELLATION.PY 


    def get_x_dense(self,nPts):
        """
        TODO: it seems the flow code has some bug with the endpoints.
        So for now I took them out.
        
        Remark: surely I had some reason for this... the points are not between
        0 and 1; rather, they are between XMINS[0] and self.XMAXS
        """
        x = np.zeros([nPts,1])
        x[:,0]=np.linspace(self.XMINS[0],self.XMAXS[0],nPts+2)[1:-1]
        x = CpuGpuArray(x)
        return x

    def get_x_dense_with_the_last_point(self,nPts):
        """
        TODO: it seems the flow code has some bug with the endpoints.
        So I here I exclude the first point, 
        and pray that including the end point will be ok. 
        """
        x = np.zeros([nPts,1])
        x[:,0]=np.linspace(self.XMINS[0],self.XMAXS[0],nPts+1)[1:]
        x = CpuGpuArray(x)
        return x

    def get_x_dense_with_both_endpoints(self,nPts):
        """
        It seems the flow code has some bug with the endpoints.
        So HBD.
        """
        x = np.zeros([nPts,1])
        x[:,0]=np.linspace(self.XMINS[0],self.XMAXS[0],nPts) 
        x = CpuGpuArray(x)
        return x
            
    def __repr__(self):
        s = "cpa space:"
        s += '\n\tCells: {}'.format(self.nC)
        s += '\n\td: {}  D: {}'.format(self.d,self.D)
        if any(self.zero_v_across_bdry):
            s += '\n\tzero bdry cond: True'
        return s        




if __name__ == '__main__':  
    import pylab 
    from pylab import plt              
    from cpab.cpa1d.needful_things import *
    
    from of.gpu import CpuGpuArray
    
    class TF:
        plot = 1 and 1
    
    params_flow_int = get_params_flow_int()
    
    
#    
#    1/0    
    
    pylab.ion()
    plt.close('all')    
 

    np.random.seed(32)

    XMINS = [0]
    XMAXS = [1]
    nCx = 10
    nCx = 50
 
    nCx = 100
#    nCx = 100*2
#    nCx = 500*2
    
    vol_preserve = False
    warp_around,zero_v_across_bdry= [False], [True]
 
     
    Ngrids=[1000]
    
    cpa_calcs=CpaCalcs(XMINS=XMINS,XMAXS=XMAXS,Ngrids=Ngrids,use_GPU_if_possible=True)    
    cpa_space=CpaSpace(XMINS,XMAXS,[nCx],zero_v_across_bdry,vol_preserve,warp_around=warp_around,
                       cpa_calcs=cpa_calcs)
    del cpa_calcs
    print cpa_space
    
    cpa_covs = CpaCovs(cpa_space,scale_spatial=(1.0) * 10, scale_value=2,
                           left_blk_rel_scale=1,
                                        right_vec_scale=1)
    
    mu = cpa_simple_mean(cpa_space)         
    Avees = cpa_space.theta2Avees(np.random.multivariate_normal(mean=mu,cov=cpa_covs.cpa_cov))             
    As=cpa_space.Avees2As(Avees)      
  
    cpa_space.update_pat(Avees)
     
    plt.close('all')
    if TF.plot:
        plt.figure()



    params_flow_int_ref = copy.deepcopy(params_flow_int)
    
    N = int(params_flow_int_ref.nTimeSteps) * 1
    
    # in general, this doesn't have to evenly spaced. Just in the right range.
    x_dense=cpa_space.get_x_dense(nPts=1000) 
    # This needs to be evenly spaced. 
    interval = np.linspace(-3,3,x_dense.size)
    
    
    Nvals = range(N)    
    Nvals=Nvals[-1:]
    
#    x_dense = CpuGpuArray(x_dense)
    v_dense = CpuGpuArray.empty_like(x_dense)
    src = x_dense
    transformed = CpuGpuArray.empty_like(src) 
    for i,N in enumerate(Nvals):        
        print 'i =',i
        params_flow_int.nTimeSteps = float(N+1)            
        if i == 0:                          
            cpa_space.calc_v(pts=src,out=v_dense)
            
           
        
       
        
#        for j in range(1):
#            if j % 100 == 0:
#                print j
#            cpa_space.calc_T(pat,pts = src, mysign=1,out=transformed,
#                           **params_flow_int)
        print 'nPts =',len(src)
        M=1
        tic=time.clock() 
        for j in range(M):
            cpa_space.calc_T_fwd(pts = src,out=transformed,
                               **params_flow_int)
         
        toc = time.clock()
        print "time (includes cpu2gpu)",(toc-tic)/M
       
        if 0: 
            # Conclusion: the difference is very small. 
            # But maybe it will important for the likelihood.
            
            transformed_gpu = gpuarray.to_gpu(np.zeros_like(src))
            
            tic=time.clock() 
            for j in range(M):
                cpa_space.calc_T(pat,pts = src, mysign=1,out=transformed_gpu,
                               **params_flow_int)
                               
            toc = time.clock()

            print "time (keeps result in gpu)",(toc-tic)/M                           
            if M==1:
                print """
                Note that for running it only once, we pay some overhead --
                probably because of the addiitonal krnl for bdry checks.
                This levels out when we make many runs"""
                
                 

                               
            if not np.allclose(transformed_gpu.get(),transformed):
                raise ValueError
            transformed = transformed_gpu.get()
         
       
       
        if TF.plot: 
            v_dense.gpu2cpu()
            transformed.gpu2cpu()
            Visualize.simple(x_dense,v_dense,interval,src,transformed,subplot_layout=[2,2])
            
    
    if computer.has_good_gpu_card:
        raw_input('raw_input:')
        print "Bye now"
