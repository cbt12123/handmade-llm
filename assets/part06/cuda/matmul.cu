// Educational row-major GEMM: naive, 16x16 shared tile, and actual cuBLAS.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define BLAS(c) do{if((c)!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error\n");exit(1);}}while(0)

__global__ void naive(const float* a,const float* b,float* c,int m,int n,int k) {
    int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*16+threadIdx.x;
    if(row<m&&col<n){float sum=0;for(int i=0;i<k;++i)sum+=a[row*k+i]*b[i*n+col];c[row*n+col]=sum;}
}

__global__ void tiled(const float* a,const float* b,float* c,int m,int n,int k) {
    __shared__ float sa[16][16],sb[16][16];
    int ty=threadIdx.y,tx=threadIdx.x,row=blockIdx.y*16+ty,col=blockIdx.x*16+tx;
    float sum=0;
    for(int base=0;base<k;base+=16){
        sa[ty][tx]=(row<m&&base+tx<k)?a[row*k+base+tx]:0;
        sb[ty][tx]=(base+ty<k&&col<n)?b[(base+ty)*n+col]:0;
        __syncthreads();
        for(int i=0;i<16;++i)sum+=sa[ty][i]*sb[i][tx];
        __syncthreads(); // No thread exits before all tile reads finish.
    }
    if(row<m&&col<n)c[row*n+col]=sum;
}

float run(int which,const float* a,const float* b,float* c,int m,int n,int k,cublasHandle_t handle,int repeats) {
    const float alpha=1,beta=0;
    dim3 block(16,16),grid((n+15)/16,(m+15)/16);
    auto launch=[&](){
        if(which==0)naive<<<grid,block>>>(a,b,c,m,n,k);
        else if(which==1)tiled<<<grid,block>>>(a,b,c,m,n,k);
        else BLAS(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,n,a,k,&beta,c,n));
    };
    for(int i=0;i<10;++i)launch();CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));CHECK(cudaEventRecord(start));
    for(int i=0;i<repeats;++i)launch();CHECK(cudaGetLastError());CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));
    float ms;CHECK(cudaEventElapsedTime(&ms,start,end));CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));return ms/repeats;
}

int main(int argc,char** argv) {
    bool small=argc>1;int m=small?23:512,n=small?19:512,k=small?37:512;
    std::vector<float>a(m*k),b(k*n),reference(m*n),result(m*n);
    for(int i=0;i<m*k;++i)a[i]=(i%17-8)*0.03f;
    for(int i=0;i<k*n;++i)b[i]=(i%13-6)*0.04f;
    float *da,*db,*dc;CHECK(cudaMalloc(&da,m*k*sizeof(float)));CHECK(cudaMalloc(&db,k*n*sizeof(float)));CHECK(cudaMalloc(&dc,m*n*sizeof(float)));
    CHECK(cudaMemcpy(da,a.data(),m*k*sizeof(float),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),k*n*sizeof(float),cudaMemcpyHostToDevice));
    cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    float blas_ms=run(2,da,db,dc,m,n,k,handle,100);CHECK(cudaMemcpy(reference.data(),dc,m*n*sizeof(float),cudaMemcpyDeviceToHost));
    float naive_ms=run(0,da,db,dc,m,n,k,handle,100);CHECK(cudaMemcpy(result.data(),dc,m*n*sizeof(float),cudaMemcpyDeviceToHost));
    float naive_error=0;for(int i=0;i<m*n;++i)naive_error=fmaxf(naive_error,fabsf(result[i]-reference[i]));
    float tiled_ms=run(1,da,db,dc,m,n,k,handle,100);CHECK(cudaMemcpy(result.data(),dc,m*n*sizeof(float),cudaMemcpyDeviceToHost));
    float tiled_error=0;for(int i=0;i<m*n;++i)tiled_error=fmaxf(tiled_error,fabsf(result[i]-reference[i]));
    printf("{\"shape\":[%d,%d,%d],\"naive_ms\":%.9g,\"tiled_ms\":%.9g,\"cublas_ms\":%.9g,\"naive_max_error\":%.9g,\"tiled_max_error\":%.9g}\n",m,n,k,naive_ms,tiled_ms,blas_ms,naive_error,tiled_error);
    BLAS(cublasDestroy(handle));CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(dc));
    return fmaxf(naive_error,tiled_error)<1e-3f?0:2;
}
