#pragma once

#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace at {

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  CHECK(!t.has(DispatchKey::Checkpoint));
  auto ret = t.add(DispatchKey::Checkpoint);
  return ret;
}

struct TORCH_API CheckpointTensorImpl : TensorImpl {
  Tensor t;
  CheckpointTensorImpl(const Tensor& t) : TensorImpl(convert_key_set(t.key_set()), t.dtype(), t.optional_device()), t(t) {
  }
};

}
