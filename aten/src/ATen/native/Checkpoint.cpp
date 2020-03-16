#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t.detach()));
}

Tensor decheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  return cpti->t;
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor checkpoint_add(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  return checkpoint(at::add(decheckpoint(a), decheckpoint(b), c));
}

}}
