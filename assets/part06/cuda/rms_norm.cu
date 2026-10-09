// FP32 educational CUDA RMSNorm. All warps share the final scale explicitly.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)

__device__ float warp_sum(float value) {
    for(int offset=16;offset>0;offset/=2) value+=__shfl_down_sync(0xffffffff,value,offset);
    return value; // Only lane zero is used as the warp result.
}

__global__ void rms(const float* x,const float* weight,float* out,int width,float eps) {
    __shared__ float partial[8]; // Launch uses exactly 256 threads: 8 complete warps.
    __shared__ float inverse;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x;
    float total=0;
    for(int column=threadIdx.x;column<width;column+=blockDim.x) {
        float value=x[row*width+column];total+=value*value;
    }
    total=warp_sum(total);
    if(lane==0) partial[warp]=total;
    __syncthreads();
    if(warp==0) {
        float sum=warp_sum(lane<8?partial[lane]:0.0f);
        if(lane==0) inverse=rsqrtf(sum/width+eps);
    }
    __syncthreads(); // Every warp must see the same final inverse.
    for(int column=threadIdx.x;column<width;column+=blockDim.x)
        out[row*width+column]=x[row*width+column]*inverse*weight[column];
}

int main() {
    const int rows=17,width=1537;const float eps=1e-6f;
    std::vector<float> x(rows*width),weight(width),out(rows*width);
    for(int i=0;i<rows*width;++i)x[i]=(i%43-21)*0.1f;
    for(int i=0;i<width;++i)weight[i]=1.0f+(i%7)*0.01f;
    float *dx,*dw,*dy;
    CHECK(cudaMalloc(&dx,x.size()*sizeof(float)));CHECK(cudaMalloc(&dw,width*sizeof(float)));CHECK(cudaMalloc(&dy,x.size()*sizeof(float)));
    CHECK(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw,weight.data(),width*sizeof(float),cudaMemcpyHostToDevice));
    rms<<<rows,256>>>(dx,dw,dy,width,eps);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(),dy,x.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double error=0;
    for(int row=0;row<rows;++row){
        double sum=0;for(int col=0;col<width;++col)sum+=double(x[row*width+col])*x[row*width+col];
        double inverse=1.0/sqrt(sum/width+eps);
        for(int col=0;col<width;++col)error=fmax(error,fabs(out[row*width+col]-x[row*width+col]*inverse*weight[col]));
    }
    printf("{\"rows\":%d,\"width\":%d,\"max_error_vs_fp64\":%.9g,\"all_warps_share_scale\":true}\n",rows,width,error);
    CHECK(cudaFree(dx));CHECK(cudaFree(dw));CHECK(cudaFree(dy));
    return error<2e-5?0:2;
}
