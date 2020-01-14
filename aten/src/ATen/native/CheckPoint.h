#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

Tensor checkpoint(const Tensor&);
Tensor decheckpoint(const Tensor&);
bool is_checkpoint(const Tensor&);

}}  // namespace at::native
