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

__device__ inline void A_times_b(double x[], double A[], double b[])
{
// Result is computed inside x. 
#if DIM == 1
  x[0] = A[0]*b[0] + A[1];
#elif DIM == 2
  x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
  x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
  
#elif DIM == 3
  //x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[3] + A[3];
  //x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[3] + A[7];
  //x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[3] + A[11];
 
  // Bug fix (Oren Freifeld, 06/23/2014)
  x[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2] + A[3];
  x[1] = A[4]*b[0] + A[5]*b[1] + A[6]*b[2] + A[7];
  x[2] = A[8]*b[0] + A[9]*b[1] + A[10]*b[2] + A[11];
  
 

  return;
#else
    return false;
#endif


//#pragma unroll
//  for(int i=0; i<DIM; ++i)
//  {
//    x[i] = A[i*(DIM+1)+DIM];
//#pragma unroll
//    for(int j=0; j<DIM; ++j)
//    {
//      x[i] += A[j+i*(DIM+1)] * b[j];
//    }
//  }
};

 

 

__device__ inline int compute_cell_idx(double* p,
                                       int nC0, int nC1, int nC2,  
                                       double inc_x,double inc_y, double inc_z)
{
int cell_idx=0;

#if DIM == 1
    //return (p[0] - fmod(p[0] , inc_x))/inc_x ;
    cell_idx =  round(min(double(nC0-1),max(0.0,(p[0] - fmod(p[0] , inc_x))/inc_x)));
    return cell_idx;
#elif DIM == 2
    {     
    if (TESS_TYPE == 2){
        // Things you learn the hard way... 
        // Must use round, since floor can give junk, depending
        // on the value of inc_x or inc_y
        cell_idx = round(min(double(nC0-1),max(0.0,(p[0] - fmod(p[0] , inc_x))/inc_x))) + 
                   round(min(double(nC1-1),max(0.0,(p[1] - fmod(p[1] , inc_y))/inc_y))) * nC0;         
        
    }
    //if (TESS_TYPE == 1)
    else
    {
        
         double p0 = min((nC0*inc_x-0.0000000001),max(0.0,p[0])) ;
         double p1 = min((nC1*inc_y-0.0000000001),max(0.0,p[1])) ;

        // BAD IDEA: This fails. 
        //double p0 = min(((nC0-1)*inc_x),max(0.0,p[0])) ;
        //double p1 = min(((nC1-1)*inc_y),max(0.0,p[1])) ;

        double xmod = fmod(p0,inc_x);
        double ymod = fmod(p1,inc_y);
        double x = xmod/inc_x ;
        double y = ymod/inc_y ;
        
 
        // We already took care of case negative values. 
        // But for values that are too high we still need to check
        // since above we used nC0 and nC1, and not nC0-1 and nC1-1.

        cell_idx = round(min(double(nC0-1),((p0 - xmod)/inc_x))) + 
                   round(min(double(nC1-1),((p1 - ymod)/inc_y))) * nC0;

  

    
        cell_idx *=4;      // every rect consists of 4 triangles
/*
                Recall the order of triangles is 
                       0
                     3   1
                       2
*/               

        // Out of bounds (left)
        if (p[0]<=0){           
            if (p[1]<=0 && p[1]/inc_y<p[0]/inc_x){  
                // Nothing to do here. 
                //cell_idx += 0;              
            }          
            else if (p[1]>=nC1*inc_y && p[1]/inc_y-nC1>-p[0]/inc_x){
                cell_idx += 2;              
            }          
            else{            
                cell_idx += 3;                     
            }
            return cell_idx;
        }        
 
        // Out of bounds (right)
        if (p[0]>=nC0*inc_x){
            if (p[1]<=0 && -p[1]/inc_y>p[0]/inc_x-nC0){                  
                cell_idx += 0;              
            }          
            else if (p[1]>=nC1*inc_y && p[1]/inc_y-nC1>p[0]/inc_x-nC0){
                 cell_idx += 2;              
            }          
            else{            
                cell_idx += 1;                     
            }
            return cell_idx; 
        }  
 
        // Out of bounds (up)
        if (p[1]<=0){            
            return cell_idx;
        }
       
        // Out of bounds (bottom)
        if (p[1]>=nC1*inc_y){
            cell_idx+=2;
            return cell_idx;
        }


        if (x<y){
            if  (1-x<y) {
                cell_idx+=2;
                }
            else {
                cell_idx+=3;
                 }
        }   
      
        else if (1-x<y) {
            cell_idx+=1;
        }        
        /* This does nothing... I leave it for clarity
        else {                   
                 cell_idx+=0;
             }
        */
        
    }
    return cell_idx;
    } 
#elif DIM == 3
    if (TESS_TYPE == 2){
      cell_idx = round(min(double(nC0-1),max(0.0,(p[0] - fmod(p[0] , inc_x))/inc_x))) + 
                 round(min(double(nC1-1),max(0.0,(p[1] - fmod(p[1] , inc_y))/inc_y))) * nC0 + 
                 round(min(double(nC2-1),max(0.0,(p[2] - fmod(p[2] , inc_z))/inc_z))) * nC1*nC0 ;     
    }
    else
    {        
        double p0 = min((nC0*inc_x-0.0000000001),max(0.0,p[0])) ;
        double p1 = min((nC1*inc_y-0.0000000001),max(0.0,p[1])) ;       
        double p2 = min((nC2*inc_x-0.0000000001),max(0.0,p[2])) ;  

        double xmod = fmod(p0,inc_x);
        double ymod = fmod(p1,inc_y);
        double zmod = fmod(p2,inc_z);         

        // We already too care of case negative values. 
        // But for values that are too high we still need to check
        // since above we used nC0 and nC1, and not nC0-1 and nC1-1.
        int i = round(min(double(nC0-1),((p0 - xmod)/inc_x)));
        int j = round(min(double(nC1-1),((p1 - ymod)/inc_y)));
        int k = round(min(double(nC2-1),((p2 - zmod)/inc_z)));
        cell_idx = i + 
              j * nC0 + 
              k * nC1 * nC0;
    
        cell_idx *=5;      // every rect consists of 5 tetra hedra

        // Need to adjust the value from above 
        // (reason: avoid issues with the circularity)
 
        // Now bring it [0,1] range
        double x = xmod/inc_x;
        double y = ymod/inc_y;
        double z = zmod/inc_z;         

        bool tf = false;
                        
        if (k%2==0){
            if ((i%2==0 && j%2==1) ||  (i%2==1 && j%2==0)){
                tf = true;
            }
        }
        else if((i%2==0 && j%2==0) ||  (i%2==1 && j%2==1)){
            tf = true;
        }              
 
        if (tf){
            double tmp = x;
            x = y;
            y = 1-tmp;
        }
                         
        // Let ti be the index of the tetrahedron. 
        // ti = 0 is the centeral.        
        // For ti = 1:
        //     -x -y +z  >= 0   (it passes thrugh the origin; no offset)
        // For ti = 2:
        //      x+y+z - 2 >= 0
        // For ti = 3:
        //      -x+y-z  >= 0       (it passes thrugh the origin; no offset)
        // For ti = 4:
        //      x-y-z >= 0         (it passes thrugh the origin; no offset)

        if (-x -y +z  >= 0){
            cell_idx+=1;
        }
        else if (x+y+z - 2 >= 0){
            cell_idx+=2;
        }
        else if (-x+y-z >= 0){
            cell_idx+=3;
        }
        else if (x-y-z >= 0){
            cell_idx+=4;
        }        
        // if none of the conditions above was triggered:
        // It is in the central tetrahedron. So "cell_idx+=0;".
            
                       
    } 

    return cell_idx;
#else
    return 0;
#endif
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
  const int nStepsOdeSolver,  const int nC0,  const int nC1,  const int nC2,
                 const double inc_x, const double inc_y,  const double inc_z)
{
   //modifies p
   double v[DIM];
   double pMid[DIM];

   int cell_idx;

  for(int t=0; t<nStepsOdeSolver; ++t)
  {
    cell_idx = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z);    
    int mi = cell_idx*DIM*(DIM+1); // index of As         
    // compute at the current location
    A_times_b(v,As+mi,p);    
    // compute mid point 
#pragma unroll
    for(int i=0; i<DIM; ++i){           
        pMid[i] = p[i] + h*v[i]/2.;        
    }
    // compute velocity at mid point
    A_times_b(v,As+mi,pMid);
    
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
  const int nStepsOdeSolver,  const int nC0,  const int nC1,  const int nC2,
                 const double inc_x, const double inc_y,  const double inc_z)
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
   double A_times_dTdAlpha[DIM]; 

   
  

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
    cell_idx = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z);    
    int mi = cell_idx*nEntries; // index of As   
    
    // compute at the current location
    A_times_b(v,As+mi,p);     
    // compute mid point 
#pragma unroll
    for(int i=0; i<DIM; ++i){           
        pMid[i] = p[i] + h*v[i]/2.;        
    }
    // compute velocity at mid point
    A_times_b(vMid,As+mi,pMid);



    
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
        A_times_b(B_times_T,Bs+ bi   , p); 
        // Term2
        A_times_b(A_times_dTdAlpha,As+mi  , q); 
        // Sum both terms 
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            u[i] = B_times_T[i] +  A_times_dTdAlpha[i] ;   
        }
        
        // Step 2: Compute mid "point"
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            qMid[i] = q[i] + h*u[i]/2.;        
        }

        // Step 3:  compute uMid

        // Term1
        A_times_b(B_times_T,Bs+ bi  , pMid);        
        // Term2
        A_times_b(A_times_dTdAlpha,As+mi  , qMid); 
        // Sum both terms 
#pragma unroll
        for(int i=0; i<DIM; ++i){           
            uMid[i] = B_times_T[i] +  A_times_dTdAlpha[i] ;   
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
             const int nPts,const int nC0, const int nC1, const int nC2, 
             double inc_x,double inc_y,double inc_z){
  //int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x*blockDim.x; 

  // Do we still need the command below?
  __syncthreads();


  if(idx >= nPts)
    return;   

  double p[DIM];  
#pragma unroll
  for(int i=0; i<DIM; ++i)
      p[i] = pts[idx*DIM+i];

  
  cell_idx[idx] = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z);                       
       
}         



__global__ void calc_T(const double* pos0,double* pos ,const double* Trels, const double* As, 
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts , const int nC0, const int nC1, const int nC2,
         const double inc_x,const double inc_y, const double inc_z)
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
      cell_idx = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z);                
      A_times_b(pNew,Trels_ + cell_idx*DIM*(DIM+1),p);
      cell_idx_new = compute_cell_idx(pNew,nC0,nC1,nC2,inc_x,inc_y,inc_z);
      if (cell_idx_new == cell_idx){
        // great, we didn't leave the cell
#pragma unroll
        for(int i=0; i<DIM; ++i){                 
                p[i] = pNew[i];
        } 
      }
      else{
        // compute using ODE solver
        solveODE(p, As_, h, nStepsOdeSolver,nC0,nC1,nC2,inc_x,inc_y,inc_z);
      }               
    } 
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}



__global__ void calc_T_simple(const double* pos0,double* pos , const double* As, 
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts , const int nC0, const int nC1, const int nC2,
         const double inc_x,const double inc_y, const double inc_z)
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
                nC0,nC1,nC2,inc_x,inc_y,inc_z);   
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}

 
__global__ void calc_grad_alpha(const double* pos0,double* pos , 
  const double* As, 
  double* Bs,
  double* grad_per_point, // shape: (nPts,dim_range,d=len(BasMats)),
  const int d,
  const double dt, const int nTimeSteps, const int nStepsOdeSolver, 
        const int nPts , const int nC0, const int nC1, const int nC2,
         const double inc_x,const double inc_y, const double inc_z)
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
                nC0,nC1,nC2,inc_x,inc_y,inc_z);   
 
#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[idx*DIM+i] = p[i];
  }
}















__global__ void calc_trajectory(double* pos, 
    const double* Trels, const double* As, double dt, int nTimeSteps, int nStepsOdeSolver,
    const int nPts,const int nC0,const int nC1,const int nC2,
    const double inc_x, const double inc_y, const double inc_z)
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
      
      cell_idx = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z);               
      A_times_b(pNew,Trels_ + cell_idx*DIM*(DIM+1),p);
      cell_idx_new = compute_cell_idx(pNew,nC0,nC1,nC2,inc_x,inc_y,inc_z);
      if (cell_idx_new == cell_idx){
        // great, we didn't leave the cell. So we can use pNew.
#pragma unroll
        for(int i=0; i<DIM; ++i){                 
            p[i] = pNew[i];
        } 
      }
      else{// We stepped outside the cell. So discard pNew
        // and compute using ODE solver instead.
        solveODE(p, As_, h, nStepsOdeSolver,nC0,nC1,nC2,inc_x,inc_y,inc_z);
      } 

#pragma unroll
    for(int i=0; i<DIM; ++i)
      pos[(idx+t*nPts)*DIM+i] = p[i];
    } 
  }
}




__global__ void calc_v(double* pos, double* vel,
                     double* As, int nPts,int nC0,int nC1,int nC2,double inc_x,
                     double inc_y, double inc_z)
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

    cell_idx = compute_cell_idx(p,nC0,nC1,nC2,inc_x,inc_y,inc_z); 
    
    A_times_b(v,As_ + cell_idx*DIM*(DIM+1),p);   

    // debug
    //v[0]=0;
    //v[1]=As[cell_idx*6+5];
    //v[1]=cell_idx==8;


#pragma unroll
    for(int i=0; i<DIM; ++i){
        vel[idx*DIM+i] = v[i];
        //vel[idx*DIM+i]= cell_idx; 
    }

  
  }
}
  









