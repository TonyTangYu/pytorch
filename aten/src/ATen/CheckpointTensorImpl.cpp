#include <ATen/CheckpointTensorImpl.h>

namespace at {

namespace native {

Tensor checkpoint(const Tensor& t) {
  throw;
}

Tensor uncheckpoint(const Tensor& t) {
  throw;
}

void pin(Tensor& t) {
  throw;
}

Tensor decheckpoint(const Tensor& t) {
  throw;
}

bool is_checkpoint(const Tensor& t) {
  throw;
}


Tensor try_checkpoint(const Tensor& t) {
  throw;
}

}

}
