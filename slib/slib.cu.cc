// --Ayan Chakrabarti <ayan@wustl.edu>
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

typedef float float32;
typedef unsigned int uint32;


__global__ void cen_kern(uint32* lhs, const float32* img, int bsz, int ht, int wd, int nc) {

  int b, y, x, c, x2, y2; uint32 out; float ctr;
  float32 * flhs = (float32 *)lhs;
  
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz*ht*wd; i += blockDim.x * gridDim.x) {
    b = i; x=b%wd;b/=wd;y=b%ht;b/=ht;
    b = b*wd*ht*nc;

    out = 0; ctr = img[i*nc];
    for(y2 = y-2; y2<=y+2; y2++)
      for(x2 = x-2; x2<=x+2; x2++)
	if(x2 != x || y2 != y) {
	  if(x2 >= 0 && x2 < wd && y2 >= 0 && y2 < ht)
	    if(ctr >= img[b+y2*wd*nc+x2*nc])
	      out |= 1;
	  out <<= 1;
	}
    
    lhs[i*nc] = out;
    for(c = 1; c < nc; c++) flhs[i*nc+c] = img[i*nc+c];
  }
}

void census(const GPUDevice& d, uint32* lhs, const float32 *img,
	    int bsz, int ht, int wd, int nc) {

  CudaLaunchConfig config = GetCudaLaunchConfig(bsz*ht*wd, d);
  cen_kern<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
    (lhs,img,bsz,ht,wd,nc);
}

__global__ void hamm_kern(float32* cv, const uint32 *left, const uint32 *right,
			  int bsz, int iht, int iwd, int y0, int x0, int ht, int wd, int nc, int dmax) {

  int b,y,x,c,d,imx,imy,lx,rx;

  const float32 *fleft = (const float32 *)left;
  const float32 *fright = (const float32 *)right;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz*ht*wd*dmax; i += blockDim.x * gridDim.x) {
    b=i; d=b%dmax;b/=dmax; //c=b%nc;b/=nc;
    x=b%wd;b/=wd;y=b%ht;b/=ht;

    imx = x+x0; imy=y+y0;
    
    if(imx >= 0 && imx < iwd && imy >= 0 && imy < iht) {
      rx = imx >= d ? imx-d : 0; lx = rx+d;
      cv[i*nc] = (float32) __popc(left[nc*(lx+iwd*(imy+iht*b))] ^ right[nc*(rx+iwd*(imy+iht*b))]);
      for(c = 1; c < nc; c++)
	cv[i*nc+c] = fabsf(fleft[c+nc*(lx+iwd*(imy+iht*b))] - fright[c+nc*(rx+iwd*(imy+iht*b))]);
    } else
      for(c = 0; c < nc; c++) cv[i*nc+c] = 0.;
  }
}

void hamm(const GPUDevice& d, float32* cv,
	  const uint32 *left, const uint32 *right,
	  int bsz, int iht, int iwd, int y0, int x0, int ht, int wd, int nc, int dmax) {

  CudaLaunchConfig config = GetCudaLaunchConfig(bsz*ht*wd*dmax, d);
  hamm_kern<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
    (cv,left,right,bsz,iht,iwd,y0,x0,ht,wd,nc,dmax);

}

#endif


