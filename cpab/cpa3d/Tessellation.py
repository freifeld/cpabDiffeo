#!/usr/bin/env python
"""
Created on Mon Mar  7 11:48:11 2016

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from of.utils import ipshell
#from scipy import sparse

from cpab.cpaNd import Tessellation as  TessellationNd

class Tessellation(TessellationNd):
    dim_domain = 3
    _LargeNumber = 10**6
    def __init__(self,nCx,nCy,nCz,nC,XMINS,XMAXS,tess):
        nCx=int(nCx)
        nCy=int(nCy)
        nCz=int(nCz)
        nC=int(nC)
        self.nCx=nCx
        self.nCy=nCy
        self.nCz=nCz
        self.nC=nC
        if len(XMINS)!=self.dim_domain:
            raise ValueError(XMINS)
        if len(XMAXS)!=self.dim_domain:
            raise ValueError(XMAXS) 
        self.XMINS=XMINS
        self.XMAXS=XMAXS
        self.type=tess
       
        cells_multiidx,cells_verts_homo_coo=self._create_cells_homo_coo(nCx,nCy,nC,tess=tess)  
        self.cells_multiidx = cells_multiidx
        self.cells_verts_homo_coo = cells_verts_homo_coo            
        if self.type=='I':
            # Recall the first tetrehedron in each 5-tuple is the central one. 
            # Its centroid correposnds to the centeroid of the hyperrectangle.
            self.box_centers=self.cells_verts_homo_coo[::5].mean(axis=1)
        elif self.type=='II':
            self.box_centers=self.cells_verts_homo_coo.mean(axis=1)
        else:
            raise ValueError(tess)

        if not isinstance(self.cells_verts_homo_coo,np.ndarray):
            raise TypeError(type(self.cells_verts_homo_coo),'expected np.ndarray')
        _xmins=self.cells_verts_homo_coo[:,:,0].min(axis=1)
        _ymins=self.cells_verts_homo_coo[:,:,1].min(axis=1)
        _zmins=self.cells_verts_homo_coo[:,:,2].min(axis=1)
        _xmaxs=self.cells_verts_homo_coo[:,:,0].max(axis=1)
        _ymaxs=self.cells_verts_homo_coo[:,:,1].max(axis=1)
        _zmaxs=self.cells_verts_homo_coo[:,:,2].max(axis=1)
        self._xmins = np.asarray(zip(_xmins,_ymins,_zmins))
        self._xmaxs = np.asarray(zip(_xmaxs,_ymaxs,_zmaxs))  

        self._xmins_LargeNumber = np.asarray(self._xmins).copy()  
        self._xmaxs_LargeNumber = np.asarray(self._xmaxs).copy()  
        self._xmins_LargeNumber[self._xmins_LargeNumber<=self.XMINS]=-self._LargeNumber
        self._xmaxs_LargeNumber[self._xmaxs_LargeNumber>=self.XMAXS]=+self._LargeNumber 
       
    def _create_cells_homo_coo(self,nCx,nCy,nC,tess='II'):
        xmin,ymin,zmin = self.XMINS
        xmax,ymax,zmax = self.XMAXS     
        tess=self.type
        nCx=self.nCx
        nCy=self.nCy
        nCz=self.nCz
        nC=self.nC
        if tess not in ['II','I']:
            raise ValueError        
        
        
              
        if tess == 'II':
            if nCz*nCy*nCx != nC:
                raise ValueError(tess,nCx,nCy,nCz,nC)
        elif tess == 'I':
            if nCz*nCy*nCx*5 != nC:
                raise ValueError(tess,nCx,nCy,nCz,5,nC)            
        else:
            raise ValueError(tess)
        
      
        Vx = np.linspace(xmin,xmax,nCx+1)
        Vy = np.linspace(ymin,ymax,nCy+1)   
        Vz = np.linspace(zmin,zmax,nCz+1)   
    
        cells_verts=[]
        cells_multiidx=[]
        
        if tess == 'II':
            for k in range(nCz):
                for i in range(nCy):
                    for j in range(nCx):   
                    
                        cells_multiidx.append((j,i,k))
                        ul0 = [Vx[j],Vy[i],Vz[k],1]
                        ur0 = [Vx[j+1],Vy[i],Vz[k],1]
                        ll0 = [Vx[j],Vy[i+1],Vz[k],1]
                        lr0 = [Vx[j+1],Vy[i+1],Vz[k],1]
                        
                        ul0 = tuple(ul0)
                        ur0 = tuple(ur0)
                        ll0 = tuple(ll0)
                        lr0 = tuple(lr0)        
                    
                        ul1 = [Vx[j],Vy[i],Vz[k+1],1]
                        ur1 = [Vx[j+1],Vy[i],Vz[k+1],1]
                        ll1 = [Vx[j],Vy[i+1],Vz[k+1],1]
                        lr1 = [Vx[j+1],Vy[i+1],Vz[k+1],1]
                        
                        ul1 = tuple(ul1)
                        ur1 = tuple(ur1)
                        ll1 = tuple(ll1)
                        lr1 = tuple(lr1)  
                
                        cells_verts.append((ul0,ur0,lr0,ll0,ul1,ur1,lr1,ll1))
        elif tess == 'I':
    
            for k in range(nCz):
                for i in range(nCy):
                    for j in range(nCx):  
                        
                        for l in range(5):
                            cells_multiidx.append((j,i,k,l))
                            
    
                                                          
                        ul0 = [Vx[j],Vy[i],Vz[k],1]
                        ur0 = [Vx[j+1],Vy[i],Vz[k],1]
                        ll0 = [Vx[j],Vy[i+1],Vz[k],1]
                        lr0 = [Vx[j+1],Vy[i+1],Vz[k],1]
                        
                        ul0 = tuple(ul0)
                        ur0 = tuple(ur0)
                        ll0 = tuple(ll0)
                        lr0 = tuple(lr0)        
                    
                        ul1 = [Vx[j],Vy[i],Vz[k+1],1]
                        ur1 = [Vx[j+1],Vy[i],Vz[k+1],1]
                        ll1 = [Vx[j],Vy[i+1],Vz[k+1],1]
                        lr1 = [Vx[j+1],Vy[i+1],Vz[k+1],1]
                        
                        ul1 = tuple(ul1)
                        ur1 = tuple(ur1)
                        ll1 = tuple(ll1)
                        lr1 = tuple(lr1)                        
    
                        tf=False                    
                        if k%2==0:
                            if (i%2==0 and j%2==1) or  (i%2==1 and j%2==0):
                                tf=True
                        else:
                            if (i%2==0 and j%2==0) or  (i%2==1 and j%2==1):
                                tf=True
                        
                        if tf:
                            ul0,ur0,lr0,ll0 = ur0,lr0,ll0,ul0
                            ul1,ur1,lr1,ll1 = ur1,lr1,ll1,ul1
                             
                                
                            
                         
                                  
                        # Beaucse of operations done later on in other files:
                        # order of the cells  matters! (I am sure)
                        # order matters of the vertices matters! (I think...)
                        cells_verts.append((ll1,ur1,ul0,lr0))  # central part
                        cells_verts.append((ul1,ur1,ll1,ul0))
                        
                        cells_verts.append((lr1,ur1,ll1,lr0))
                        cells_verts.append((ll0,ul0,lr0,ll1))
                        cells_verts.append((ur0,ul0,lr0,ur1))
                        
                        
                        
                       
        else:
            raise ValueError(tess)
    
    #    ipshell('hi')
    #    1/0
    
        if len(cells_multiidx) != nC:
            raise ValueError( len(cells_multiidx) , nC)        
        if len(cells_verts) != nC:
            raise ValueError( len(cells_verts) , nC)   
    
        if tess == 'II':    
            # every cell should be made of 8 vertices (cube)                 
            if not all([x==8 for x in map(len,map(set,cells_verts))]):         
                raise ValueError        
        elif tess == 'I':    
            # every cell should be made of 4 vertices (tetrahedron)                 
            if not all([x==4 for x in map(len,map(set,cells_verts))]):         
                raise ValueError
        
        cells_verts =np.asarray(cells_verts) 
        return  cells_multiidx,cells_verts

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
        nCz=self.nCz
        
        if valid_outside:
            raise NotImplementedError('dim_domain =',dim_domain,
                                      'valid_outside =',valid_outside)
        if dim_domain !=3:
            raise ValueError(dim_domain)
        
        nbrs = np.zeros((nC,nC))
        
        mi=cells_multiidx # shorter name
        for i in range(nC):
            for j in range(nC):
                # shorter names
                mi = cells_multiidx[i]
                mj = cells_multiidx[j]
                
                # tetrahedron index within the box
                ti = mi[-1]
                tj = mj[-1]
                if len(mi)!=4:
                    raise ValueError(len(mi))
                if len(mj)!=4:
                    raise ValueError(len(mj))
                  
                vi = cells_verts[i]
                vj = cells_verts[j]

                vi=self.make_it_hashable(vi)
                vj=self.make_it_hashable(vj)      
     
                if mi == mj:
                    continue
                elif mi[:-1]==mj[:-1]:
                    # Same rect boxs, different tetrahedra
                    if ti==0 or tj==0: 
                        if tj==ti:
                            raise ValueError
                        else:                                        
                            nbrs[i,j]=1
                            
                else:
                   # Different boxes            
                    if len(set(vi).intersection(vj))==3:                                
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
    
         
        
        verts1 = []
        verts2 = []    
        verts3 = []   
        verts4 = []
        counter = 0  
        for h in H:
            if h.any():  
    #        if h.nnz:
            
                # Very annoying: I think there is a bug in the sparse matrix object.
                # Even after 'todense' it is impossible to flatten it properly.            
    #            h = np.asarray(h.todense().tolist()[0])  # Workaround.
                
                 
                
                # Find the vertex pair
                i = (h==1).nonzero()[0][0]     
                j = (h==-1).nonzero()[0][0]
                
                mi = cells_multiidx[i] # for debugging
                mj = cells_multiidx[j] # for debugging
                ti = mi[-1]
                tj = mj[-1]                
                
    #            a = mi
    #            b = mj
                
                vi = cells_verts[i]
                vj = cells_verts[j]
                           
                vi=self.make_it_hashable(vi)
                vj=self.make_it_hashable(vj)

                side = set(vi).intersection(vj)
                len_side = len(side)
                if len_side == 3:                 
                    v1,v2,v3 = np.asarray(list(side))
                    v4=None
                                    
            
                    verts1.append(v1)
                    verts2.append(v2)
                    verts3.append(v3)
                    verts4.append(v4)
                elif len_side==2:
                    if ti == 0 and tj == 0:
                        # That's ok. Can ignore it:
                        # these should be the two "Central" tetrahedra 
                        # of two adjacent cell. 
                        continue
                    else:
                        raise ValueError
                elif len_side == 1:
                    continue
                    # I thinkg this should be ok.
                    # I don't have time now to check this happens only when 
                    # it should... TODO
                else:
                    print ('len(side) = ',len(side))
                    ipshell('wtf')
                    raise ValueError(len(side),side)                
                
                counter+=len_side
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
    
        # Every side connects 3 vertices. 
        nPtsInSide = 3  
    
        
        if counter != nSides*nPtsInSide:
            ipshell('WTF')
            raise ValueError(counter,nSides)      
        
       
        # At every vertex, all components of the velocity must agree.
     
        nConstraints = nSides*nPtsInSide*dim_range 
     
        verts1 = np.asarray(verts1)
        verts2 = np.asarray(verts2)   
        verts3 = np.asarray(verts3) 
        verts4 = np.asarray(verts4) 
    
        H = np.asarray([h for h in H if h.any()])    
    #    H = np.asarray([h for h in H if h.nnz])                                   
     
    
    
    #    
     #    ipshell('hi')
    #    1/0 
        
        
        return verts1,verts2,verts3,verts4,H,nSides,nConstraints    




    def create_verts_and_H_type_II(self,dim_range):
        """
        This assummes 3D 
        
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
#        nCz=self.nCz
        
        if dim_domain !=3:
            raise ValueError(dim_domain)
        if dim_range != dim_domain:
            raise NotImplementedError(dim_range)
#        if dim_range not in (1,dim_domain):
#            raise NotImplementedError(dim_range) 
        
        nbrs = np.zeros((nC,nC))
        
        mi=cells_multiidx # shorter name
        for i in range(nC):
            for j in range(nC):
                if mi[i] == mi[j]:
                    continue
                else:
    #               pair = (np.abs(mi[i][0]-mi[j][0]),
    #                       np.abs(mi[i][1]-mi[j][1]))
    #               if set(pair) == set([0,1]):
    #                   nbrs[i,j]=1
                   triplet = (np.abs(mi[i][0]-mi[j][0]),
                              np.abs(mi[i][1]-mi[j][1]),
                              np.abs(mi[i][2]-mi[j][2]))
                   triplet=np.asarray(triplet)
                   if (triplet==0).sum()==2 and (triplet==1).sum()==1:
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
    
         
        
        verts1 = []
        verts2 = []    
        verts3 = []   
        verts4 = []
        counter = 0  
        for h in H:
            if h.any():  
    #        if h.nnz:
            
                # Very annoying: I think there is a bug in the sparse matrix object.
                # Even after 'todense' it is impossible to flatten it properly.            
    #            h = np.asarray(h.todense().tolist()[0])  # Workaround.
                
                 
                counter+=4
                # Find the vertex pair
                i = (h==1).nonzero()[0][0]     
                j = (h==-1).nonzero()[0][0]
    #            a = mi[i]
    #            b = mi[j]
                
                vi = cells_verts[i]
                vj = cells_verts[j]
                           
                vi = self.make_it_hashable(vi)
                vj = self.make_it_hashable(vj)
                
                side = set(vi).intersection(vj)
                if len(side) != 4: # adjcant boxes share 4 verts
                    ipshell('oops')
                    raise ValueError(len(side),side)
                try:
                    v1,v2,v3,v4 = np.asarray(list(side))
                except:
                    ipshell('hi')
                    raise                
        
                verts1.append(v1)
                verts2.append(v2)
                verts3.append(v3)
                verts4.append(v4)
        #        if a != (1,1):
        #            continue
        #        print a, ' is a nbr of ',b
        
        if counter != nSides*4:
            raise ValueError(counter,nEdges)      
         
        # Every side conntect 4 vertices. 
        # At every vertex, all components of the velocity must agree.
    #    nConstraints = nSides*4*dim_domain
        nConstraints = nSides*4*dim_range 
     
        verts1 = np.asarray(verts1)
        verts2 = np.asarray(verts2)   
        verts3 = np.asarray(verts3) 
        verts4 = np.asarray(verts4) 
    
        H = np.asarray([h for h in H if h.any()])    
    #    H = np.asarray([h for h in H if h.nnz])                                   
     
    
    
    #    
     #    ipshell('hi')
    #    1/0 
        
        
        return verts1,verts2,verts3,verts4,H,nSides,nConstraints    
    
    
    
    
    
    







        

    def create_constraint_mat_bdry(self,
                      zero_v_across_bdry,
                      verbose=False):

        dim_domain=self.dim_domain
        nC = self.nC
        cells_verts=self.cells_verts_homo_coo
        
        if dim_domain != 3:
            raise ValueError(self.dim_domain)
        if len(zero_v_across_bdry)!=3:
            raise ValueError(zero_v_across_bdry)
        zero_vx_across_bdry,zero_vy_across_bdry,zero_vz_across_bdry = zero_v_across_bdry
        
        xmin,ymin,zmin = self.XMINS
        xmax,ymax,zmax = self.XMAXS
        
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
                if zero_vz_across_bdry and v[2] in (zmin,zmax):
                    if verbose:
                        print 'vx', ' cell',i , 'vert ', j
                    L.append(np.roll(row,nHomoCoo*2)) 
                           
        L = np.asarray(L)
        
        return L

    def create_constraint_mat_bdry_separable(self,
                      zero_v_across_bdry,
                      verbose=False):
        raise NotImplementedError("""
        If the condition was zero velocity at the boundary, this would have been trivial to code.
        But till now we used only "zero normal component at the boundary".
        So while it is separable, it is slightly more messy. TODO.
        """
        )
        dim_domain=self.dim_domain
        nC = self.nC
        cells_verts=self.cells_verts_homo_coo
        
        if dim_domain != 3:
            raise ValueError(self.dim_domain)
        if len(zero_v_across_bdry)!=3:
            raise ValueError(zero_v_across_bdry)
        zero_vx_across_bdry,zero_vy_across_bdry,zero_vz_across_bdry = zero_v_across_bdry
        
        xmin,ymin,zmin = self.XMINS
        xmax,ymax,zmax = self.XMAXS
        
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
                if zero_vz_across_bdry and v[2] in (zmin,zmax):
                    if verbose:
                        print 'vx', ' cell',i , 'vert ', j
                    L.append(np.roll(row,nHomoCoo*2)) 
                           
        L = np.asarray(L)
        
        return L
