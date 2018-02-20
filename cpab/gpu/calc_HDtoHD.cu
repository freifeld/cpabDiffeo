/*Created on Mon Feb  10 10:00:00 2014



Oren Freifeld
Email: freifeld@csail.mit.edu
*/

 

#ifndef DIM
  #define DIM 2
#endif

#ifndef TESS_TYPE
  #define TESS_TYPE 2
#endif

__device__ inline void A_times_b_affine(double x[], double A[], double b[])
{
// Result is computed inside x. 
#pragma unroll
  for(int i=0; i<DIM; ++i)
  {
    x[i] = A[i*(DIM+1)+DIM];
#pragma unroll
    for(int j=0; j<DIM; ++j)
    {
      x[i] += A[i*(DIM+1)+j] * b[j];
    }
  }
};

__device__ inline void A_times_b_linear(double x[], double A[], double b[])
{
// Result is computed inside x. 
#pragma unroll
  for(int i=0; i<DIM; ++i)
  {
	x[i]=0;  
    //x[i] = A[i*(DIM+1)+DIM];  // Offset 
#pragma unroll
    for(int j=0; j<DIM; ++j)
    {
      x[i] += A[i*(DIM+1)+j] * b[j];
    }
  }
}; 

 

__device__ inline int compute_cell_idx(double* p,
                                       const int* nCs,
                                       const double* incs
                                       )
{
	int cell_idx=0;
    /*
	cell_idx = round(min(double(nC0-1),max(0.0,(p[0] - fmod(p[0] , inc_x))/inc_x))) + 
			   round(min(double(nC1-1),max(0.0,(p[1] - fmod(p[1] , inc_y))/inc_y))) * nC0 + 
			   round(min(double(nC2-1),max(0.0,(p[2] - fmod(p[2] , inc_z))/inc_z))) * nC1*nC0 ;     
	*/
    //int i=0;
    int c=1;
    
#pragma unroll
    for(int i=0; i<DIM; ++i)
    {
      cell_idx += c * round(min(double(nCs[i]-1),max(0.0,(p[i] - fmod(p[i] , incs[i]))/incs[i])));
      c *= nCs[i];
      
    }   
     
    

    return cell_idx;

};


 
__device__ inline bool inBoundaryCoo(double *p, double *bbs,int coo)
{    return bbs[coo*2] <= p[coo] && p[coo] < bbs[coo*2+1] ;
};


__device__ inline bool inBoundary(double *p, double *bbs)
{
#if DIM == 1
    return inBoundaryCoo(p, bbs,0);
#elif DIM == 2
    return inBoundaryCoo(p, bbs,0) && inBoundaryCoo(p, bbs,1);
#elif DIM == 3
    return inBoundaryCoo(p, bbs,0) && 
           inBoundaryCoo(p, bbs,1) && 
           inBoundaryCoo(p, bbs,2);
#else
    return false;
#endif
}


__device__ void solveODE(double *p,  double* As, const double h, 
  const int nStepsOdeSolver, 
                 const int* nCs,
                 const double* incs
                 )
{
   //modifies p
   double v[DIM];
   double pMid[DIM];

   int cell_idx;

  for(int t=0; t<nStepsOdeSolver; ++t)
  {
    cell_idx = compute_cell_idx(p,nCs,incs);    
    int mi = cell_idx*DIM*(DIM+1); // index of As         
    // compute at the current location
    A_times_b_affine(v,As+mi,p);    
    // compute mid point 
#pragma unroll
    for(int i=0; i<DIM; ++i){           
        pMid[i] = p[i] + h*v[i]/2.;        
    }
    // compute velocity at mid point
    A_times_b_affine(v,As+mi,pMid);
    
    // update p 
#pragma unroll
    for(int i=0; i<DIM; ++i){
        p[i] += v[i]*h; 
    }           
  }
}




__device__ void solveODE2(double *p,  double* As,  double* Bs,
  double* grad_per_point, // shape: (nPts,dim_range,d=len(BasMats)),
  int idx,
  int d,
  int nPts,
  const double h, 
  const int nStepsOdeSolver, 
  const int* nCs,
  const double* incs
  )
{
   //modifies p
   double v[DIM];
   double pMid[DIM];
   double vMid[DIM];

   double q[DIM]; 
   double qMid[DIM]; 
   double u[DIM];
   double uMid[DIM]; 

   double B_times_T[DIM]; 
   double A_times_dTdtheta[DIM]; 

   
  

   int cell_idx;
   int nEntries = DIM*(DIM+1);

    // set to zero

    for (int j=0; j<d; j++){
#pragma unroll
        for(int i=0; i<DIM; ++i){
                 // nPts,dim_range,d 
                grad_per_point[idx*DIM*d + i * d + j] = 0;
        }
    }

  for(int t=0; t<nStepsOdeSolver; ++t)
  {
    cell_idx = compute_cell_idx(p,nCs,incs);    
    int mi = cell_idx*nEntries; // index of As   
    
    // compute at the current location
    A_times_b_affine(v,As+mi,p);     
    // compute mid point 
#pragma unroll
    for(int i=0; i<DIM; ++i){           
        pMid[i] = p[i] + h*v[i]/2.;        
    }
    // compute velocity at mid point
    A_times_b_affine(vMid,As+mi,pMid);



    
    for (int j=0; j<d; j++){
        int bi = j * nEntries*N_CELLS +  mi ; // index of the Bs
        

        // copy q
#pragma unroll
        for(int i=0; i<DIM; ++i){
             // nPts,dim_range,d 
            q[i] =  grad_per_point[idx*DIM*d + i * d + j];
        }  

    
        // Step 1: Compute u using the old location

        // Find current RHS (term1 + term2)
        // Term1
        A_times_b_affine(B_times_T,Bs+ bi   , p); 
        // Term2
        A_times_b_linear(A_times_dTdtheta,As+mi  , q); 
        // Sum both terms 
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            u[i] = B_times_T[i] +  A_times_dTdtheta[i] ;   
        }
        
        // Step 2: Compute mid "point"
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            qMid[i] = q[i] + h*u[i]/2.;        
        }

        // Step 3:  compute uMid

        // Term1
        A_times_b_affine(B_times_T,Bs+ bi  , pMid);        
        // Term2
        A_times_b_linear(A_times_dTdtheta,As+mi  , qMid); 
        // Sum both terms 
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            uMid[i] = B_times_T[i] +  A_times_dTdtheta[i] ;   
        } 
        
        // update q
#pragma unroll
        for(int i=0; i<DIM; ++i){
            q[i] += uMid[i]*h; 
        }    
        // 
#pragma unroll
        for(int i=0; i<DIM; ++i){
             // nPts,dim_range,d 
            grad_per_point[idx*DIM*d + i * d + j] = q[i];


             
        }  


    }



      

    
    // update p 
#pragma unroll
    for(int i=0; i<DIM; ++i){
        p[i] += vMid[i]*h; 
    }           
  }
}











__global__ void calc_cell_idx(double* pts,
             int* cell_idx,
             const int nPts,
             const int* nCs,
             const double* incs
             ){

  int idx = threadIdx.x + blockIdx.x*blockDim.x; 

  // Do we still need the command below?
  __syncthreads();


  if(idx >= nPts)
    return;   

  double p[DIM];  
#pragma unroll
  for(int i=0; i<DIM; ++i){
      p[i] = pts[idx*DIM+i];
  }
 
  cell_idx[idx] = compute_cell_idx(p,nCs,incs);                       
       
}         



__global__ void calc_T(const double* pos0,double* pos ,const double* Trels, const double* As, 
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts ,
         const int* nCs,
         const double* incs
         )
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 
      
  __shared__ double Trels_[N_CELLS*DIM*(DIM+1)]; 
  __shared__ double As_[N_CELLS*DIM*(DIM+1)];
   
  
  if(tid < N_CELLS)
  {
// copy from GPU RAM into grid-cell shared memory
 
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i){
      Trels_[i] = Trels[i];
//#pragma unroll
//   for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i)
      As_[i] = As[i];
    }
  }
 

  __syncthreads();
  if(idx < nPts)
  {
    double p[DIM];  
    double pNew[DIM];
#pragma unroll
    for(int i=0; i<DIM; ++i)
    {
      pos[idx*DIM+i]=pos0[idx*DIM+i]; // copy the initial location
      p[i] = pos[idx*DIM+i];
    }
    double h = dt/double(nStepsOdeSolver);
    int cell_idx=0;
    int cell_idx_new =0;    

    for (int t=0; t<nTimeSteps; ++t)
    {
      cell_idx = compute_cell_idx(p,nCs,incs);                
      A_times_b_affine(pNew,Trels_ + cell_idx*DIM*(DIM+1),p);
      cell_idx_new = compute_cell_idx(pNew,nCs,incs);
      if (cell_idx_new == cell_idx){
        // great, we didn't leave the cell
#pragma unroll
        for(int i=0; i<DIM; ++i){                 
                p[i] = pNew[i];
        } 
      }
      else{
        // compute using ODE solver
        solveODE(p, As_, h, nStepsOdeSolver,nCs,incs);
      }               
    } 
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}



__global__ void calc_T_simple(const double* pos0,double* pos , const double* As, 
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts , 
        //const int nC0, const int nC1, const int nC2,
        // const double inc_x,const double inc_y, const double inc_z
        int* nCs,double* incs
        )
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 
      
   __shared__ double As_[N_CELLS*DIM*(DIM+1)];      
  if(tid < N_CELLS)
  {
// copy from GPU RAM into grid-cell shared memory  
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i)
      As_[i] = As[i];
  } 
  __syncthreads();

  if(idx < nPts)
  {
    double p[DIM];      
#pragma unroll
    for(int i=0; i<DIM; ++i)
    {
      p[i]=pos0[idx*DIM+i]; // copy the initial location      
    }
    double h = dt/double(nStepsOdeSolver);    
    solveODE(p, As_, h, nStepsOdeSolver * nTimeSteps,
                //nC0,nC1,nC2,inc_x,inc_y,inc_z
                nCs,incs
                );   
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}

 
__global__ void calc_grad_theta(const double* pos0,double* pos , 
  const double* As, 
  double* Bs,
  double* grad_per_point, // shape: (nPts,dim_range,d=len(BasMats)),
  const int d,
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts , 
        //const int nC0, const int nC1, const int nC2,
         //const double inc_x,const double inc_y, const double inc_z
         int* nCs,double* incs
         )
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 
      
   __shared__ double As_[N_CELLS*DIM*(DIM+1)];      
  if(tid < N_CELLS)
  {
// copy from GPU RAM into grid-cell shared memory  
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i)
      As_[i] = As[i];
  } 
  __syncthreads();

  if(idx < nPts)
  {
    double p[DIM];      
#pragma unroll
    for(int i=0; i<DIM; ++i)
    {
      p[i]=pos0[idx*DIM+i]; // copy the initial location      
    }
    double h = dt/double(nStepsOdeSolver);    
    solveODE2(p, As_, Bs,
              grad_per_point,
              idx,     
              d,   
              nPts,       
              h, nStepsOdeSolver * nTimeSteps,
              nCs,incs);   
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}















__global__ void calc_trajectory(double* pos, 
    const double* Trels, const double* As, double dt, int nTimeSteps, int nStepsOdeSolver,
    const int nPts,
    //const int nC0,const int nC1,const int nC2,
    //const double inc_x, const double inc_y, const double inc_z
    int* nCs,double* incs
    )
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 
  
 
  __shared__ double Trels_[N_CELLS*DIM*(DIM+1)];
  __shared__ double As_[N_CELLS*DIM*(DIM+1)];
  
  if(tid < N_CELLS)
  { 
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i)
      Trels_[i] = Trels[i];
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); ++i)
      As_[i] = As[i];
  }
  

  __syncthreads();
  if(idx < nPts)
  {
    double p[DIM];  
    double pNew[DIM];
#pragma unroll
    for(int i=0; i<DIM; ++i){
      p[i] = pos[idx*DIM+i]; // copy initial location
    }
    double h = dt/double(nStepsOdeSolver);
    int cell_idx=0;
    int cell_idx_new =0;

    for (int t=0; t<nTimeSteps; ++t)
    {
      
      cell_idx = compute_cell_idx(p,nCs,incs);               
      A_times_b_affine(pNew,Trels_ + cell_idx*DIM*(DIM+1),p);
      cell_idx_new = compute_cell_idx(pNew,nCs,incs);
      if (cell_idx_new == cell_idx){
        // great, we didn't leave the cell. So we can use pNew.
#pragma unroll
        for(int i=0; i<DIM; ++i){                 
            p[i] = pNew[i];
        } 
      }
      else{// We stepped outside the cell. So discard pNew
        // and compute using ODE solver instead.
        solveODE(p, As_, h, nStepsOdeSolver,nCs,incs);
      } 

#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[(idx+t*nPts)*DIM+i] = p[i];
    } 
  }
}




__global__ void calc_v(double* pos, double* vel,
                     double* As, int nPts,
                     //int nC0,int nC1,int nC2,double inc_x,
                    // double inc_y, double inc_z
                     int* nCs,double* incs
                     )
{
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 
 
 
  __shared__ double As_[N_CELLS*DIM*(DIM+1)];
  
  if(tid < N_CELLS)
  {
// copy from GPU RAM into grid-cell shared memory
#pragma unroll
    for(int i=tid*DIM*(DIM+1); i<(tid+1)*DIM*(DIM+1); i++)
      As_[i] = As[i];
  }
  __syncthreads();
  
  if(idx < nPts)
  {
    int cell_idx=0;
    double p[DIM];    
    double v[DIM];
 
#pragma unroll
    for(int i=0; i<DIM; ++i){
      p[i] = pos[idx*DIM+i];
      v[i] = vel[idx*DIM+i];
    }

    cell_idx = compute_cell_idx(p,nCs,incs); 
    

    A_times_b_affine(v,As_ + cell_idx*DIM*(DIM+1),p);   

     


#pragma unroll
    for(int i=0; i<DIM; ++i){
        vel[idx*DIM+i] = v[i]; 
//vel[idx*DIM+i]=cell_idx;

    }

  
  }
}
  









