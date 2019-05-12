// --Ayan Chakrabarti <ayan@wustl.edu>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

typedef unsigned int uint32;
typedef float float32;

// Compute Census
void census(const GPUDevice& d, uint32* lhs, const float32 *img,
	    int bsz, int ht, int wd, int nc);

class censusGPU : public OpKernel {
public:
  explicit censusGPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& img_ = context->input(0);
    Tensor * lhs_;
    TensorShape imshp = img_.shape();

    OP_REQUIRES_OK(context,context->allocate_output(0,imshp,&lhs_));


    const float32 * img; uint32 *lhs;
    int bsz,ht,wd,nc;

    bsz = imshp.dim_size(0); ht = imshp.dim_size(1); wd = imshp.dim_size(2); nc = imshp.dim_size(3);
      
    img = img_.flat<float32>().data();
    lhs = (uint32*) lhs_->flat<float32>().data();
    
    census(context->eigen_device<GPUDevice>(),lhs,img,bsz,ht,wd,nc);
  }
};


REGISTER_OP("Census")
.Input("img: float32")
.Output("out: float32")
.SetShapeFn(shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("Census").Device(DEVICE_GPU), censusGPU);


inline int getsc(OpKernelContext * ctx, int id) {
  const Tensor& t = ctx->input(id); return t.scalar<int32>()();
}

// Compute Hamming
void hamm(const GPUDevice& d, float32* cv,
	  const uint32 *left, const uint32 *right,
	  int bsz, int iht, int iwd, int y0, int x0, int ht, int wd, int nc, int dmax);

class hammGPU : public OpKernel {
public:
  explicit hammGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,context->GetAttr("dmax", &dmax));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& left_ = context->input(0);
    const uint32 *left = (const uint32*)left_.flat<float32>().data();

    const Tensor& right_ = context->input(1);
    const uint32 *right = (const uint32*)right_.flat<float32>().data();

    TensorShape ishp = left_.shape();

    int bsz,iht,iwd, nc;
    bsz = ishp.dim_size(0); iht = ishp.dim_size(1); iwd = ishp.dim_size(2); nc = ishp.dim_size(3);


    int y0 = getsc(context,2), x0 = getsc(context,3),
      ht = getsc(context,4), wd = getsc(context,5);

    Tensor * cv_;
    OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape({bsz,ht,wd,dmax,nc}),&cv_));
      
    float32 * cv = cv_->flat<float32>().data();
    
    hamm(context->eigen_device<GPUDevice>(),cv,left,right,bsz,iht,iwd,y0,x0,ht,wd,nc,dmax);
  }
private:
  int dmax;
};


REGISTER_OP("Hamming")
.Input("left: float32")
.Input("right: float32")
.Input("y: int32")
.Input("x: int32")
.Input("ht: int32")
.Input("wd: int32")
.Attr("dmax: int")
.Output("cv: float32")
.SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("Hamming").Device(DEVICE_GPU)
			.HostMemory("x").HostMemory("y")
			.HostMemory("wd").HostMemory("ht")
			,hammGPU);
