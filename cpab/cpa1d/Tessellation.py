#!/usr/bin/env python
"""
Created on Mon Mar  7 13:04:26 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
class Tessellation(object):   
    dim_domain = 1
    _LargeNumber = 10**6    
    def __init__(self,nCx,XMINS,XMAXS):
        self.type='I'
        nC = nCx  # of cells
        self.nC=nC 
        self.nCx=nCx
        self.XMINS=XMINS
        self.XMAXS=XMAXS
        
        
        cells_multiidx,cells_verts_homo_coo=self.create_cells_homo_coo(nCx,self.XMINS,self.XMAXS) 
        self.cells_multiidx = cells_multiidx
        self.cells_verts_homo_coo = cells_verts_homo_coo
        
        self.box_centers=self.cells_verts_homo_coo.mean(axis=1)  


        endpoints = self.cells_verts_homo_coo[:,:,:-1].reshape(self.nC,-1)
        self._xmins =np.asarray(endpoints[:,0])
        self._xmaxs =np.asarray(endpoints[:,1])
        self._xmins = self._xmins[:,np.newaxis] # need it to be a 2d array with one column.
        self._xmaxs = self._xmaxs[:,np.newaxis] 
        
        self._xmins_LargeNumber = self._xmins.copy()  
        self._xmaxs_LargeNumber = self._xmaxs.copy()  
        self._xmins_LargeNumber[self._xmins_LargeNumber<=self.XMINS]=-self._LargeNumber
        self._xmaxs_LargeNumber[self._xmaxs_LargeNumber>=self.XMAXS]=+self._LargeNumber 
        
        

    def create_cells_homo_coo(self,nCx,XMINS,XMAXS):
        xmin=XMINS[0]
        xmax=XMAXS[0]
        nC = nCx
        Vx = np.linspace(xmin,xmax,nCx+1)
        cells_x = []
        cells_x_verts = [] 
        nC=int(nC)
        for i in range(nC):              
            cells_x.append([i])
            l = [Vx[i],1]
            r = [Vx[i+1],1]               
            l = tuple(l)
            r = tuple(r)             
            cells_x_verts.append((l,r))  

        cells_multiidx,cells_verts = cells_x,cells_x_verts  
        cells_verts =np.asarray(cells_verts) 
        return  cells_multiidx,cells_verts 




    def create_verts_and_H(self):  
        nC = self.nC
        cells_x = self.cells_multiidx
        cells_x_verts = self.cells_verts_homo_coo
        dim_domain = self.dim_domain
        if dim_domain !=1:
            raise NotImplementedError(dim_domain)
        nbrs = np.zeros((nC,nC))
        for i in range(nC):
            for j in  range(nC):
                if cells_x[i] == cells_x[j]:
                    continue
                else:
                   
                   singelton = [np.abs(cells_x[i][0]-cells_x[j][0])]
                               
                   if set(singelton) == set([1]):
                       nbrs[i,j]=1
                                         
        
        
        nEdges=nbrs.sum().astype(np.int)/2
    
        H = np.zeros((nC**2,nC))
    #    H = sparse.lil_matrix((nC**2,nC))
        
        for i in range(nC):
            for j in range(nC):        
                k = i*nC + j
                if i < j:
                    continue
                nbr = nbrs[i,j]
                if nbr:
                    H[k,i]=-1
                    H[k,j]=+1
    
           
        
        verts1 = []
        k = 0
        for h in H:
    #        ipshell('..')        
            if h.any():  
    #        if h.nnz:
            
                # Very annoying: I think there is a bug in the sparse matrix object.
                # Even after 'todense' it is impossible to flatten it properly.            
    #            h = np.asarray(h.todense().tolist()[0])  # Workaround.
                
                 
                k+=2
                i = (h==1).nonzero()[0][0]     
                j = (h==-1).nonzero()[0][0]
    #            a = cells_xy[i]
    #            b = cells_xy[j]
                
                vi = cells_x_verts[i]
                vj = cells_x_verts[j]
               
               # TypeError: unhashable type: 'numpy.ndarray'
                def make_it_hashable(arr):
                   return tuple([tuple(r.tolist()) for r in arr])
                   
                vi=make_it_hashable(vi)
                vj=make_it_hashable(vj)
#                raise ValueError(vi)
               
                edge = set(vi).intersection(vj)

               
               
                try:
                    v1 = np.asarray(list(edge))
                except:
                    ipshell('hi')
                    raise                
        
                verts1.append(v1)
                
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
        
        if k != nEdges*2:
            raise ValueError(k,nEdges)      
            
         
        # At every shared vertex, the velocities must agree.
        nConstraints = nEdges*dim_domain
         
     
        verts1 = np.asarray(verts1)
       
        H = np.asarray([h for h in H if h.any()])    
    #    H = np.asarray([h for h in H if h.nnz])                                   
     
        
        return verts1,H,nEdges,nConstraints    
    #    return verts1,verts2,H,nConstraints   
    
    
    
    

    def create_constraint_mat_bdry(self,
                      zero_v_across_bdry,
                      verbose=False):
        
        XMINS=self.XMINS
        XMAXS=self.XMAXS
        cells_verts_homo_coo=self.cells_verts_homo_coo
        nC=self.nC
        dim_domain=self.dim_domain
        xmin=XMINS[0]
        xmax=XMAXS[0]
        if dim_domain != 1:
            raise ValueError
        nHomoCoo = dim_domain+1
        length_Avee = dim_domain*nHomoCoo
        nCols = nC*length_Avee
        
        L = [] 
        for i,cell in enumerate(cells_verts_homo_coo):
            for j,v in enumerate(cell):            
                # s stands for start
                # e stands for end
                
                s = i*length_Avee 
                e = s+nHomoCoo
                row = np.zeros(nCols)
                row[s:e] = v
                
                
                if zero_v_across_bdry[0] and v[0] in (xmin,xmax):
                    if verbose:
                        print 'vx', ' cell',i , 'vert ', j
                    L.append(row)                       
               
                            
        L = np.asarray(L)
        
        return L







if __name__ == "__main__":
    pass
