#!/usr/bin/env python
"""
Created on Mon Mar 28 11:00:45 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from itertools import product # since we have an unknown number of lists
from numpy import binary_repr


from of.utils import ipshell
#from scipy import sparse

from cpab.cpaNd import Tessellation as  TessellationNd

class Tessellation(TessellationNd):    
    _LargeNumber = 10**6
    def __init__(self,nCs,nC,XMINS,XMAXS,tess,dim_domain,dim_range):
        if tess !='II':
            raise NotImplementedError
        if dim_domain!=dim_range:
            raise NotImplementedError
        nC=int(nC)
        nCs=np.asarray(map(int,nCs)) 
        self.nCs=nCs
        self.nC=nC
        self.dim_domain=dim_domain
        
               
        
        
        if len(XMINS)!=self.dim_domain:
            raise ValueError(XMINS)
        if len(XMAXS)!=self.dim_domain:
            raise ValueError(XMAXS) 
        self.XMINS=XMINS
        self.XMAXS=XMAXS
        self.type=tess
        
        cells_multiidx,cells_verts_homo_coo=self._create_cells_homo_coo()  
        self.cells_multiidx = cells_multiidx
        self.cells_verts_homo_coo = cells_verts_homo_coo 
                 
        if self.type=='II':
            self.box_centers=self.cells_verts_homo_coo.mean(axis=1)
        else:
            raise ValueError(tess)

        if not isinstance(self.cells_verts_homo_coo,np.ndarray):
            raise TypeError(type(self.cells_verts_homo_coo),'expected np.ndarray')
        
        N = len(nCs)
        self._xmins = zip(np.asarray([self.cells_verts_homo_coo[:,:,coo].min(axis=1) 
                       for coo in range(N)]).T)
        self._xmaxs = zip(np.asarray([self.cells_verts_homo_coo[:,:,coo].max(axis=1) 
                       for coo in range(N)]).T)
                         
        self._xmins = [tuple(x[0].tolist()) for x in self._xmins]
        self._xmaxs = [tuple(x[0].tolist()) for x in self._xmaxs]        


        if not isinstance(self.cells_verts_homo_coo,np.ndarray):
            raise TypeError(type(self.cells_verts_homo_coo),'expected np.ndarray')
        
        self._xmins_LargeNumber = np.asarray(self._xmins).copy()  
        self._xmaxs_LargeNumber = np.asarray(self._xmaxs).copy()  
        self._xmins_LargeNumber[self._xmins_LargeNumber<=self.XMINS]=-self._LargeNumber
        self._xmaxs_LargeNumber[self._xmaxs_LargeNumber>=self.XMAXS]=+self._LargeNumber                 

        
        
    def _create_cells_homo_coo(self):
        tess=self.type
        nCs=self.nCs
        nC=self.nC
        XMINS=self.XMINS
        XMAXS=self.XMAXS
        
        N = len(nCs)
        if len(XMINS)!=N:
            raise ValueError(XMINS)
        if len(XMAXS)!=N:
            raise ValueError(XMAXS) 
        
        if tess != 'II':
            raise NotImplementedError(tess)
        
         
        if np.prod(nCs) != nC:
            raise ValueError(np.prod(nCs), nCs)
                
        nCs = map(int,nCs)
        
        Vs = [np.linspace(m,M,nc+1) for (m,M,nc) in zip(XMINS,XMAXS,nCs)]
    
    
        cells_verts=[]
        cells_multiidx=[]
      
        lists = map(range,nCs)
        lists = lists[::-1]
        Vs=Vs[::-1]
        
        brs = [binary_repr(i,N) for i in range(2**N)]
        
        
        for idx, items in enumerate(product(*lists)):
    #        print idx,'---',items        
            items=np.asarray(items)   
            verts_of_this_cell =[]
            for i in range(2**N):
                inc = np.asarray(map(int,brs[i]))
                
                indices = items+inc
                
                tmp = [Vs[j][indices[j]] for j in range(N)][::-1]
                
                verts_of_this_cell.append(tmp+[1])
                
                
                
            cells_multiidx.append( tuple(items.tolist()))
            
            verts_of_this_cell = map(tuple,verts_of_this_cell)
            verts_of_this_cell = tuple(verts_of_this_cell)
            cells_verts.append(verts_of_this_cell)
                
    
    #    ipshell('hi')
    #    2/0     
       
            
       
            
        if len(cells_multiidx) != nC:
            raise ValueError( len(cells_multiidx) , nC)        
        if len(cells_verts) != nC:
            raise ValueError( len(cells_verts) , nC)   
    
        if tess == 'II':    
            # every cell should be made of 8 vertices (cube)                 
            if not all([x==2**N for x in map(len,map(set,cells_verts))]):         
                raise ValueError        
        else:
            raise NotImplementedError(tess)
        
          
        cells_verts =np.asarray(cells_verts) 
        return  cells_multiidx,cells_verts

    def create_verts_and_H_type_II(self,dim_range):
        """
    
        """
        dim_domain = self.dim_domain
        nCs = self.nCs
        nC = self.nC
        cells_multiidx=self.cells_multiidx
        cells_verts=self.cells_verts_homo_coo
        N = len(nCs)

        if dim_domain !=N:
            raise ValueError(dim_domain)
        
        nbrs = np.zeros((nC,nC))
        
        
        mi=cells_multiidx # shorter name
        for i in range(nC):
            for j in range(nC):
                if mi[i] == mi[j]:
                    continue
                else:
                     
                    
                    t = np.abs(np.asarray(mi[i])-np.asarray(mi[j]))
                   
                    
                    if (t==0).sum()==N-1 and (t==1).sum()==1:
                       nbrs[i,j]=1            
    
        
        nSides=nbrs.sum().astype(np.int)/2   
        
        
    
        
        # H is larger than we need
        # but it was easier to code this way.
        # Later we eliminate the unused rows.
        H = np.zeros((nC**2,nC))
    #    H = sparse.lil_matrix((nC**2,nC))
        
        for i in range(nC):
            for j in range(nC):   
                # k is the index of the row in H.
                # Most rows won't be used.
                k = i*nC + j
                if i < j:
                    continue
                nbr = nbrs[i,j]
                if nbr:
                    H[k,i]=-1
                    H[k,j]=+1
    
         
        
    #    verts1 = []
    #    verts2 = []    
    #    verts3 = []   
    #    verts4 = []
        
        verts = [[] for i in range(2**(N-1))]
        counter = 0  
        for h in H:
            if h.any():  
    #        if h.nnz:
            
                # Very annoying: I think there is a bug in the sparse matrix object.
                # Even after 'todense' it is impossible to flatten it properly.            
    #            h = np.asarray(h.todense().tolist()[0])  # Workaround.
            
                # update: the sparsity issue was because I used arrays
                # while the sparse functions want matrices.
                
                 
                counter+=2**(N-1)
                # Find the vertex pair
                i = (h==1).nonzero()[0][0]     
                j = (h==-1).nonzero()[0][0]
    
                
                vi = cells_verts[i]
                vj = cells_verts[j]
                
                vi=self.make_it_hashable(vi)
                vj=self.make_it_hashable(vj)
                           
                side = set(vi).intersection(vj)
                if len(side) != 2**(N-1): # adjcant boxes share 2**(N-1) verts
                    ipshell('oops')
                    raise ValueError(len(side),side)
                
                _verts = np.asarray(list(side))
                
    #            try:
    #                
    #                v1,v2,v3,v4 = np.asarray(list(side))
    #            except:
    #                ipshell('hi')
    #                raise                
        
    #            verts1.append(v1)
    #            verts2.append(v2)
    #            verts3.append(v3)
    #            verts4.append(v4)
                for i in range(2**(N-1)):
                    verts[i].append(_verts[i])
                
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
        
        if counter != nSides*2**(N-1):
            raise ValueError(counter,nSides)      
         
        # Every side conntect 2**(N-1) vertices. 
        # At every vertex, all components of the velocity must agree.
    ##############    nConstraints = nSides*2**(N-1)*dim_domain
        nConstraints = nSides*2**(N-1)*dim_range 
    # 
    #    verts1 = np.asarray(verts1)
    #    verts2 = np.asarray(verts2)   
    #    verts3 = np.asarray(verts3) 
    #    verts4 = np.asarray(verts4) 
    
        H = np.asarray([h for h in H if h.any()])    
    #    H = np.asarray([h for h in H if h.nnz])                                   
     
    
    
    #  
        
        
        return verts,H,nSides,nConstraints    