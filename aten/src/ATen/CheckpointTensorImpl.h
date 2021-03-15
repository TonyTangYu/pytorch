#pragma once

#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace at {

struct TORCH_API CheckpointTensorImpl : TensorImpl {
};

}
