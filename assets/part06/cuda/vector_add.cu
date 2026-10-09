// Standalone CUDA: odd tail, checked launches, real event timing.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void add(const float* a,const float* b,float* out,int n) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n) out[index]=a[index]+b[index];
}

int main() {
    const int n=1048579,threads=256,blocks=(n+threads-1)/threads;
    const size_t bytes=n*sizeof(float);
    std::vector<float> a(n),b(n),out(n);
    for(int i=0;i<n;++i) {a[i]=(i%97)*0.01f;b[i]=(i%31)*0.02f;}
    float *da,*db,*dc;
    CUDA_CHECK(cudaMalloc(&da,bytes));CUDA_CHECK(cudaMalloc(&db,bytes));CUDA_CHECK(cudaMalloc(&dc,bytes));
    CUDA_CHECK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    for(int i=0;i<20;++i) add<<<blocks,threads>>>(da,db,dc,n);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,end;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<200;++i) add<<<blocks,threads>>>(da,db,dc,n);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed;CUDA_CHECK(cudaEventElapsedTime(&elapsed,start,end));elapsed/=200;
    CUDA_CHECK(cudaMemcpy(out.data(),dc,bytes,cudaMemcpyDeviceToHost));
    float error=0;for(int i=0;i<n;++i) error=fmaxf(error,fabsf(out[i]-(a[i]+b[i])));
    if(error>1e-6f) return 2;
    printf("{\"n\":%d,\"blocks\":%d,\"threads\":%d,\"max_error\":%.9g,\"event_ms\":%.9g,\"effective_GB_s\":%.9g}\n",
           n,blocks,threads,error,elapsed,3.0*bytes/(elapsed*1e6));
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));
}
