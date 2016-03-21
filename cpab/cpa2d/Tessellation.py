#!/usr/bin/env python
"""
Created on Mon Mar  7 11:48:11 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""




import numpy as np
from of.utils import ipshell
from scipy import sparse

from cpab.cpaNd import Tessellation as  TessellationNd

   
class Tessellation(TessellationNd):
    dim_domain = 2
    _LargeNumber = 10**6
    def __init__(self,nCx,nCy,nC,XMINS,XMAXS,tess):
        nCx=int(nCx)
        nCy=int(nCy)
        nC=int(nC)
        self.nCx=nCx
        self.nCy=nCy
        self.nC=nC
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
        if self.type=='I':               
            # THIS IS SPECIFIC FOR TRI TESS IN 2D ONLY 
            # Recall that the first 4 triangles have one shared point:
            # The center of the first rectangle. And this point is the first
            # in each of these 4 triangles.
            # The next four triangles share the center of the second rectangtle,
            # And so on.            
            self.box_centers=self.cells_verts_homo_coo[::4][:,0].copy()
            
        elif self.type=='II':            
            self.box_centers=self.cells_verts_homo_coo.mean(axis=1) 
        else:
            raise ValueError(tess)


        _xmins=self.cells_verts_homo_coo[:,:,0].min(axis=1)
        _ymins=self.cells_verts_homo_coo[:,:,1].min(axis=1)
        _xmaxs=self.cells_verts_homo_coo[:,:,0].max(axis=1)
        _ymaxs=self.cells_verts_homo_coo[:,:,1].max(axis=1)
        self._xmins = np.asarray(zip(_xmins,_ymins))
        self._xmaxs = np.asarray(zip(_xmaxs,_ymaxs))


        self._xmins_LargeNumber = np.asarray(self._xmins).copy()  
        self._xmaxs_LargeNumber = np.asarray(self._xmaxs).copy()  
        self._xmins_LargeNumber[self._xmins_LargeNumber<=self.XMINS]=-self._LargeNumber
        self._xmaxs_LargeNumber[self._xmaxs_LargeNumber>=self.XMAXS]=+self._LargeNumber 
        
        
        self.permuted_indices = np.random.permutation(self.nC)  


    def _create_cells_homo_coo(self):
        xmin,ymin = self.XMINS
        xmax,ymax = self.XMAXS     
        tess=self.type
        nCx=self.nCx
        nCy=self.nCy
        nC=self.nC
        if tess not in ['II','I']:
            raise ValueError
     
        Vx = np.linspace(xmin,xmax,nCx+1)
        Vy = np.linspace(ymin,ymax,nCy+1)   
        cells_x = []
        cells_x_verts = [] 
        if tess == 'II':
            for i in range(nCy):
                for j in range(nCx):        
                    cells_x.append((j,i))
                    ul = [Vx[j],Vy[i],1]
                    ur = [Vx[j+1],Vy[i],1]
                    ll = [Vx[j],Vy[i+1],1]
                    lr = [Vx[j+1],Vy[i+1],1]
                    
                    ul = tuple(ul)
                    ur = tuple(ur)
                    ll = tuple(ll)
                    lr = tuple(lr)        
                    
                    cells_x_verts.append((ul,ur,lr,ll))
        elif tess == 'I':
            for i in range(nCy):
                for j in range(nCx):                                                        
                    ul = [Vx[j],Vy[i],1]
                    ur = [Vx[j+1],Vy[i],1]
                    ll = [Vx[j],Vy[i+1],1]
                    lr = [Vx[j+1],Vy[i+1],1]                
                    
                    ul = tuple(ul)
                    ur = tuple(ur)
                    ll = tuple(ll)
                    lr = tuple(lr)                        
                     
                    center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
                    center = tuple(center)                 
                    
                    cells_x_verts.append((center,ul,ur))  # order matters!
                    cells_x_verts.append((center,ur,lr))  # order matters!
                    cells_x_verts.append((center,lr,ll))  # order matters!
                    cells_x_verts.append((center,ll,ul))  # order matters!                
    
                    cells_x.append((j,i,0))
                    cells_x.append((j,i,1))
                    cells_x.append((j,i,2))
                    cells_x.append((j,i,3))
                                 
        else:
            raise NotImplementedError(tess)
        
        if len(cells_x_verts) != nC:
            raise ValueError( len(cells_x_verts) , nC)        
        if len(cells_x) != nC:
            raise ValueError( len(cells_x) , nC)         
        
        cells_multiidx,cells_verts = cells_x,cells_x_verts  
        cells_verts =np.asarray(cells_verts) 
        return  cells_multiidx,cells_verts 
    

#    def create_verts_and_H(self,dim_range,
##              nCx,nCy,nC, cells_multiidx,
##              cells_verts,dim_domain,dim_range,
#              valid_outside
#                              ):  
#        """      
#        H encodes the n'bors info.
#        """    
#        if self.type == 'I':
#            return self.create_verts_and_H_type_I(dim_range,valid_outside)
#        elif self.type=='II':
#            return self.create_verts_and_H_type_II(dim_range)
#        else:
#            raise NotImplementedError
            
            
    def create_verts_and_H_type_I(self,dim_range,valid_outside):  
        """
        This assummes 2D 
        
        H encodes the n'bors info.
        """    
        if self.type != 'I':
            raise ValueError(self.type)
        dim_domain=self.dim_domain
        nC = self.nC
        cells_multiidx=self.cells_multiidx
        cells_verts=self.cells_verts_homo_coo
        nCx=self.nCx
        nCy=self.nCy
        
        if dim_domain !=2:
            raise ValueError(dim_domain)
        if dim_range not in (1,2):
            raise NotImplementedError(dim_range) 
            
        nbrs = np.zeros((nC,nC),dtype=np.bool)
        
        if valid_outside:
            left=np.zeros((nC,nC),np.bool)    
            right=np.zeros((nC,nC),np.bool) 
            top=np.zeros((nC,nC),np.bool) 
            bottom=np.zeros((nC,nC),np.bool) 

 
        print 'Encoding continuity constraints.'
        print 'If nC is large, this may take some time.'
        print 'For a given configuration, however, this is done only once;'
        print 'the results computed here will be saved and reused the next time'
        print 'you use the same configuration.'
    
        # TODO: Cython
#        
        
        for i in range(nC):
            if nC > 200 and i % 200==0:
                print i,'/',nC
            for j in range(nC):
                # shorter names
                mi = cells_multiidx[i]
                mj = cells_multiidx[j]
                
                vi = cells_verts[i]
                vj = cells_verts[j]
                
                vi=self.make_it_hashable(vi)
                vj=self.make_it_hashable(vj)
                
                shared_verts = set(vi).intersection(vj)            
                
                if len(mi)!=3:
                    raise ValueError
                if len(mj)!=3:
                    raise ValueError
                if mi == mj:  
                    # same cell, nothing to do here
                    continue
                elif mi[:-1]==mj[:-1]:
                    # Same rect boxs, different triangles
                    s = set([mi[-1],mj[-1]])
                    if s in [set([0,1]),set([1,2]),set([2,3]),set([3,0])]:
                        nbrs[i,j]=1
                else:
                    # different rect boxes
                
    
                    if valid_outside:
        #                 try to deal with the extension
                        if mi[0]==mj[0]==0: # leftmost col
                            if mi[2]==mj[2]==3: # left triangle                     
                                if np.abs(mi[1]-mj[1])==1: # adjacent rows
                                    nbrs[i,j]=1
                                    left[i,j]=True
                                    continue
        
                        if mi[0]==mj[0]==nCx-1: # rightmost col
                            if mi[2]==mj[2]==1: # right triangle                     
                                if np.abs(mi[1]-mj[1])==1: # adjacent rows
                                    nbrs[i,j]=1
                                    right[i,j]=True
                                    continue
        
                        if mi[1]==mj[1]==0: # uppermost row
                            if mi[2]==mj[2]==0: # upper triangle                     
                                if np.abs(mi[0]-mj[0])==1: # adjacent cols
                                    nbrs[i,j]=1
                                    top[i,j]=True
                                    continue
        
                        if mi[1]==mj[1]==nCy-1: # lowermost row
                            if mi[2]==mj[2]==2: # lower triangle                     
                                if np.abs(mi[0]-mj[0])==1: # adjacent cols
                                    nbrs[i,j]=1
                                    bottom[i,j]=True
                                    continue                    
                        
                    if set([mi[2],mj[2]]) not in [set([0,2]),set([1,3])]:
                        continue
                        
                    pair = (mi[0]-mj[0]),(mi[1]-mj[1])
                    
                    
                    # Recall the order of triangles is 
                    #         0
                    #       3   1
                    #         2
                    
                    # vertical nbr's     
                    if pair == (0,1) and (mi[2],mj[2])==(0,2):
                       
                        nbrs[i,j]=1
                    elif pair == (0,-1) and (mi[2],mj[2])==(2,0):  
                        
                        nbrs[i,j]=1
                    # horizontal nbr's    
                    elif pair == (1,0) and  (mi[2],mj[2])==(3,1):    
                         
                        nbrs[i,j]=1 
                    elif pair == (-1,0) and  (mi[2],mj[2])==(1,3):    
                         
                        nbrs[i,j]=1      
                  
       
    
        print 'Creating H of size (nC**2,nC)=({},{})'.format(nC**2,nC) 
        try:  
            H = np.zeros((nC**2,nC))
        except MemoryError:
            msg='Got MemoryError when trying to call np.zeros((nC**2,nC))'
            ipshell(msg)
            raise MemoryError('np.zeros((nC**2,nC))','nC={}'.format(nC))
             
        
#        H = sparse.lil_matrix((nC**2,nC))
        
        for i in range(nC):
            for j in range(nC):        
                k = i*nC + j
                if i < j:
                    continue
                nbr = nbrs[i,j]
                if nbr:
                    H[k,i]=-1
                    H[k,j]=+1
        
    #    ipshell('save H')
    #    1/0
        verts1 = []
        verts2 = []        
        k = 0
        print 'Extracting the vertices'
        for count,h in enumerate(H):
            if H.shape[0]>1000:
                if count % 100000 == 0:
                    print count,'/',H.shape[0] 
    #        ipshell('..')     
        
            if h.any():  
#            if h.nnz:
#                ipshell('STOP')
#                # Make h dense and flat
#                h=np.asarray(h.todense()).ravel()
                                                   
                                    
                i = (h==1).nonzero()[0][0]     
                j = (h==-1).nonzero()[0][0]
    
                mi = cells_multiidx[i]
                mj = cells_multiidx[j]
                
                vi = cells_verts[i]
                vj = cells_verts[j]
               
                vi=self.make_it_hashable(vi)
                vj=self.make_it_hashable(vj)
                
                shared_verts = set(vi).intersection(vj)
                
                if len(shared_verts) ==0:
                    continue
                if len(shared_verts) ==1:                
                    # single vertex
                    if any([left[i,j],right[i,j],top[i,j],bottom[i,j]]):
                        # shared_vert is a set that contains a single tuple.                    
                        v_aux = list(shared_verts)[0] # v_aux is a tuple
                        v_aux = list(v_aux) # Now v_aux is a list (i.e. mutable)
                        if left[i,j] or right[i,j]:
                            v_aux[0]-=10 # Create a new vertex  with the same y
                        elif top[i,j] or bottom[i,j]:
                            v_aux[1]-=10 # Create a new vertex  with the same x
                        else:
                            raise ValueError("WTF?")                        
                        v_aux = tuple(v_aux)
                        shared_verts.add(v_aux) # add it to the set  
    #                    ipshell('hello')
    #                    print shared_verts
                    else:
                        # We can skip it since the continuity at this vertex 
                        # will be imposed via the edges.
                        continue 
                
                if len(shared_verts) != 2:
                    ipshell('oops')
                    raise ValueError(len(shared_verts),shared_verts)
                try:
                    v1,v2 = np.asarray(list(shared_verts))
                except:
                    ipshell('oops2')
                    raise                
                k+=2    
                verts1.append(v1)
                verts2.append(v2)
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
        
    #    nEdges=nbrs.sum().astype(np.int)/2    
    #    if k != nEdges*2:
    #        raise ValueError(k,nEdges)      
    
        nEdges = k/2   
            
        # Every edge connects 2 vertices. 
        # At every vertex, all components of the velocity must agree.
        nConstraints = nEdges*2*dim_range
        
            
     
        verts1 = np.asarray(verts1)
        verts2 = np.asarray(verts2)    
        
        
        H = np.asarray([h for h in H if h.any()])    
#        H = np.asarray([h for h in H if h.nnz])                                   
        print 'H is ready'   
         
        return verts1,verts2,H,nEdges,nConstraints    
        
        
        
    def create_verts_and_H_type_II(self,
#                                   nC, cells_multiidx, cells_verts,dim_domain,
                                   dim_range):  
        """
        This assummes 2D 
        
        H encodes the n'bors info.
        """    
        if self.type != 'II':
            raise ValueError(self.type)
        dim_domain=self.dim_domain
        nC = self.nC
        cells_multiidx=self.cells_multiidx
        cells_verts=self.cells_verts_homo_coo
#        nCx=self.nCx
#        nCy=self.nCy
        
        if dim_domain !=2:
            raise ValueError(dim_domain)
        if dim_range not in (1,dim_domain):
            raise NotImplementedError(dim_range)  
               
        nbrs = np.zeros((nC,nC))
        for i in range(nC):
            for j in range(nC):
                # shorter names
                mi = cells_multiidx[i]
                mj = cells_multiidx[j]            
                
                if mi == mj:
                    continue
                else:
                   pair = (np.abs(mi[0]-mj[0]),
                           np.abs(mi[1]-mj[1]))
                           
                   if set(pair) == set([0,1]):
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
    
    #    ipshell('hi')
    #    1/0    
        
        verts1 = []
        verts2 = []        
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
                
    #            if set([i,j])==set([6,9]):
    #                ipshell('debug')
    #                1/0
    #            a = mi
    #            b = mj
                
                vi = cells_verts[i]
                vj = cells_verts[j]
                
                vi = self.make_it_hashable(vi)
                vj = self.make_it_hashable(vj)
               
                edge = set(vi).intersection(vj)
                if len(edge) != 2:
                    ipshell('oops')
                    raise ValueError(len(edge),edge)
                try:
                    v1,v2 = np.asarray(list(edge))
                except:
                    ipshell('oops2')
                    raise                
        
                verts1.append(v1)
                verts2.append(v2)
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
        
        if k != nEdges*2:
            raise ValueError(k,nEdges)      
            
        # Every edge connects 2 vertices. 
        # At every vertex, all components of the velocity must agree.
        #nConstraints = nEdges*2*dim_domain
        nConstraints = nEdges*2*dim_range
        
     
    
    
    
     
        verts1 = np.asarray(verts1)
        verts2 = np.asarray(verts2)    
    
        H = np.asarray([h for h in H if h.any()])    
    #    H = np.asarray([h for h in H if h.nnz])                                   
      
     #    ipshell('hi')
    #    1/0 
        
        
        return verts1,verts2,H,nEdges,nConstraints    
    #    return verts1,verts2,H,nConstraints           
        
        
        
    def create_constraint_mat_bdry(self,
                      zero_v_across_bdry,
                      verbose=False):

        dim_domain=self.dim_domain
        nC = self.nC
        cells_multiidx=self.cells_multiidx
        cells_verts=self.cells_verts_homo_coo
        nCx=self.nCx
        nCy=self.nCy
        
        if dim_domain != 2:
            raise ValueError(self.dim_domain)
        if len(zero_v_across_bdry)!=2:
            raise ValueError(zero_v_across_bdry)
        zero_vx_across_bdry,zero_vy_across_bdry = zero_v_across_bdry
        
        xmin,ymin = self.XMINS
        xmax,ymax = self.XMAXS
        
        nHomoCoo = dim_domain+1
        length_Avee = dim_domain*nHomoCoo
        nCols = nC*length_Avee
        
        L = [] 
        for i,cell in enumerate(cells_verts):
            for j,v in enumerate(cell):            
                # s stands for start
                # e stands for end
                
                s = i*length_Avee 
                e = s+nHomoCoo
                row = np.zeros(nCols)
                row[s:e] = v
               
                if zero_vx_across_bdry and v[0] in (xmin,xmax):
                    if verbose:
                        print 'vx', ' cell',i , 'vert ', j
                    L.append(row)    
                     
                                  
                if zero_vy_across_bdry and v[1] in (ymin,ymax):
                    if verbose:
                        print 'vy', ' cell',i , 'vert ', j
                    L.append(np.roll(row,nHomoCoo))
                            
        L = np.asarray(L)
        
        return L
        