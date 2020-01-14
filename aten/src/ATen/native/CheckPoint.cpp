#include <ATen/native/CheckPoint.h>
#include <ATen/CheckPointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckPointTensorImpl>::make(t.detach()));
}

Tensor decheckpoint(const Tensor& t) {
  return from_tensor(t)->get();
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckPointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor checkpoint_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), vec.at(1), alpha)};
    };
  strongs s = {from_tensor(self), from_tensor(other)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_mul(const Tensor& self, const Tensor& other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), vec.at(1))};
    };
  strongs s = {from_tensor(self), from_tensor(other)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_mul(const Tensor& self, Scalar other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec[0], other)};
    };
  strongs s = {from_tensor(self)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor like_zeros(const Tensor& a) {
  return at::zeros_like(a, MemoryFormat::Preserve);
}

Tensor checkpoint_like_zeros(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::like_zeros(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor like_ones(const Tensor& a) {
  return at::ones_like(a, MemoryFormat::Preserve);
}

Tensor checkpoint_like_ones(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::like_ones(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_linear(const Tensor& a, const Tensor& b, const Tensor& c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::linear(vec[0], vec[1], vec[2])};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor_maybe(c)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_relu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::relu(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_threshold_backward(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold_backward(vec[0], vec[1], c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_tanh_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh_backward(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor select_backward(at::Tensor const& grad, c10::ArrayRef<long> input_sizes, long dim, long index) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.select(dim, index).copy_(grad);
  return grad_input;
}

Tensor checkpoint_select_backward(at::Tensor const& grad, c10::ArrayRef<long> input_sizes, long dim, long index) {
  std::vector<long> input_sizes_ = input_sizes.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select_backward(vec[0], input_sizes, dim, index)};
    };
  strongs s = {from_tensor(grad)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_gelu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::gelu(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_softmax(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_softmax(vec[0], b, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_log_softmax(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax(vec[0], b, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_addmm(const Tensor& a, const Tensor& b, const Tensor& c, Scalar d, Scalar e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addmm(vec[0], vec[1], vec[2], d, e)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_mm(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mm(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_transpose(const Tensor& a, long b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::transpose(vec[0], b, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_transpose(const Tensor& a, Dimname b, Dimname c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::transpose(vec[0], b, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_sign(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sign(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss_forward(const Tensor& a, const Tensor& b, const Tensor& c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::nll_loss_forward(vec[0], vec[1], vec[2], d, e);
      return {std::get<0>(res), std::get<1>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1]};
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckPointTensorImpl::mutate(mt, {a, b});
  return a;
}

Tensor& checkpoint_mul_(Tensor& a, const Tensor& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(vec.at(1));
    };
  CheckPointTensorImpl::mutate(mt, {a, b});
  return a;
}

Tensor& checkpoint_relu_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).relu_();
    };
  CheckPointTensorImpl::mutate(mt, {a});
  return a;
}

Tensor& checkpoint_sign_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sign_out(a_, vec.at(1));
    };
  CheckPointTensorImpl::mutate(mt, {a, b});
  return a;
}

Tensor& checkpoint_squeeze_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_();
    };
  CheckPointTensorImpl::mutate(mt, {a});
  return a;
}

Tensor& checkpoint_squeeze_(Tensor& a, Dimname b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckPointTensorImpl::mutate(mt, {a});
  return a;
}

Tensor& checkpoint_squeeze_(Tensor& a, long b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckPointTensorImpl::mutate(mt, {a});
  return a;
}

Tensor& checkpoint_zero_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).zero_();
    };
  CheckPointTensorImpl::mutate(mt, {a});
  return a;
}

Tensor checkpoint__cat(c10::ArrayRef<Tensor> a, long b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cat(vec, b)};
    };
  strongs s;
  for (const Tensor& t : a) {
    s.push_back(from_tensor(t));
  }
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor& checkpoint__cat_out(Tensor& a, c10::ArrayRef<Tensor> b, long c) {
  std::vector<Tensor> args;
  args.push_back(a);
  for (const Tensor& t : b) {
    args.push_back(t);
  }
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor t = vec[0];
      at::cat_out(t, ArrayRef<Tensor>(vec.data() + 1, vec.size() - 1), c);
    };
  CheckPointTensorImpl::mutate(mt, args);
  return a;
}

Tensor checkpoint_clone(const Tensor& a, c10::optional<c10::MemoryFormat> b) {
  //todo: check b
  return a;
}

Tensor& checkpoint_fill_(Tensor& a, Scalar b) {
  AT_ERROR("fill_");
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor a = at::clone(vec[0]);
      return {at::fill_(a, b)};
    };
  strongs s = {from_tensor(a)};
  Tensor ret = CheckPointTensorImpl::make(rt, s)[0];
  cell_from_tensor(a)->value = cell_from_tensor(ret)->value;
  return a;
}

Tensor& checkpoint_copy_(Tensor& a, const Tensor& b, bool c) {
  AT_ERROR("copy_");
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor a = at::clone(vec[0]);
      return {a.copy_(vec[1], c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  Tensor ret = CheckPointTensorImpl::make(rt, s)[0];
  cell_from_tensor(a)->value = cell_from_tensor(ret)->value;
  return a;
}

Tensor checkpoint_nll_loss_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e, long f, const Tensor& g) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nll_loss_backward(vec[0], vec[1], vec[2], vec[3], e, f, vec[4])};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor_maybe(d), from_tensor(g)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_log_softmax_backward(const Tensor& a, const Tensor& b, long c, const Tensor& d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax_backward_data(vec[0], vec[1], c, vec[2])};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(d)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_matmul(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::matmul(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_max(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_min(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::min(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_index_select(const Tensor& self, long dim, const Tensor& index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::index_select(vec[0], dim, vec[1])};
  };
  strongs s = {from_tensor(self), from_tensor(index)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor> checkpoint_prelu_backward(const Tensor& a, const Tensor& b, const Tensor& c) {
  AT_ERROR("prelu");
}

Tensor checkpoint_binary_cross_entropy(Tensor const&, Tensor const&, Tensor const&, long) {
  AT_ERROR("binary_cross_entropy");
}

Tensor& checkpoint_binary_cross_entropy_out(at::Tensor&, at::Tensor const&, at::Tensor const&, at::Tensor const&, long) {
  AT_ERROR("binary_cross_entropy_out");
}

Tensor checkpoint_sigmoid(Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_div(Tensor const& a, Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_sub(Tensor const& a, Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec[0], vec[1], c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint__s_where(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_s_where(vec[0], vec[1], vec[2])};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_kl_div(at::Tensor const& a, at::Tensor const& b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div(vec[0], vec[1], c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_kl_div_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div_backward(vec[0], vec[1], vec[2], d)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_sigmoid_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid_backward(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor& checkpoint_binary_cross_entropy_backward_out(Tensor&, Tensor const&, Tensor const&, Tensor const&, Tensor const&, long) {
  AT_ERROR("bcebackward");
}

Tensor& checkpoint_avg_pool2d_backward_out(Tensor&, Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, bool, c10::optional<long>) {
  AT_ERROR("avg");
}

Tensor checkpoint_neg(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::neg(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_abs(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_tanh(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh(vec[0])};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_avg_pool2d(Tensor const& a, IntArrayRef b, IntArrayRef c, IntArrayRef d, bool e, bool f, c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d(vec[0], b_, c_, d_, e, f, g)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_l1_loss(Tensor const&, Tensor const&, long) {
  AT_ERROR("l1_loss");
}

Tensor checkpoint_min(Tensor const&) {
  AT_ERROR("min");
}

Tensor checkpoint_avg_pool2d_backward(Tensor const& a, Tensor const& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d_backward(vec[0], vec[1], c_, d_, e_, f, g, h)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_softmax_backward(Tensor const&, Tensor const&, long, Tensor const&) {
  AT_ERROR("softmax_bwd");
}

Tensor checkpoint_prelu(Tensor const&, Tensor const&) {
  AT_ERROR("prelu");
}

Tensor checkpoint_max(Tensor const&) {
  AT_ERROR("max");
}

Tensor checkpoint_l1_loss_backward(Tensor const&, Tensor const&, Tensor const&, long) {
  AT_ERROR("l1_loss_bwd");
}

Tensor checkpoint_pow(Tensor const&, c10::Scalar) {
  AT_ERROR("pow");
}

Scalar checkpoint_local_scalar_dense(Tensor const& a) {
  return at::_local_scalar_dense(decheckpoint(a));
}

Tensor& checkpoint_div_(Tensor&, Tensor const&) {
  AT_ERROR("div_");
}

Tensor checkpoint_le(const Tensor&, Tensor const&) {
  AT_ERROR("le");
}

Tensor checkpoint_lt(const Tensor&, Tensor const&) {
  AT_ERROR("lt");
}

Tensor checkpoint_ge(const Tensor&, Tensor const&) {
  AT_ERROR("ge");
}

Tensor checkpoint_gt(const Tensor&, Tensor const&) {
  AT_ERROR("gt");
}

Tensor checkpoint_eq(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_sum(const Tensor& a, c10::ArrayRef<long> b, bool c, c10::optional<c10::ScalarType> d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec[0], b_, c, d)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_sum(const Tensor& a, c10::optional<c10::ScalarType> b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec[0], b)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_binary_cross_entropy_backward(Tensor const&, Tensor const&, Tensor const&, Tensor const&, long) {
  AT_ERROR("bce");
}

Tensor& checkpoint_l1_loss_backward_out(Tensor&, Tensor const&, Tensor const&, Tensor const&, long) {
  AT_ERROR("l1_loss_bwd");
}

Tensor checkpoint_gelu_backward(Tensor const&, Tensor const&) {
  AT_ERROR("gelu_bwd");
}

// We assume everything is purely functional as of the ICML push for simplicity.
Tensor checkpoint_view(const Tensor& a, IntArrayRef b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec[0].view(b_)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor> checkpoint__max(const Tensor& a, long b, bool c) {
  AT_ERROR("_max");
}

std::tuple<Tensor, Tensor> checkpoint_max(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::max(vec[0], b, c);
      return {std::get<0>(res), std::get<1>(res)};
    };
  strongs s = {from_tensor(a)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor> checkpoint_min(const Tensor& a, long, bool) {
  AT_ERROR("min");
}

Tensor checkpoint_cudnn_convolution_transpose_backward_weight(c10::ArrayRef<long>, Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) {
  AT_ERROR("convolution_transpose_backward_weight");
}

Tensor checkpoint_cudnn_convolution_transpose_backward_bias(Tensor const&) {
  AT_ERROR("convolution_transpose_backward_bias");
}

Tensor checkpoint_cudnn_convolution_backward_weight(c10::ArrayRef<long>, Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) {
  AT_ERROR("convolution_transpose_backward_weight");
}

Tensor checkpoint_cudnn_convolution_transpose_backward_input(Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) {
  AT_ERROR("convolution_transpose_backward_input");
}

Tensor checkpoint_cudnn_convolution(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution(vec[0], vec[1], c_, d_, e_, f, g, h)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_cudnn_convolution_backward_input(c10::ArrayRef<long>, Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) {
  AT_ERROR("convolution_backward_input");
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_backward(Tensor const& a, Tensor const& b, Tensor const& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, std::array<bool, 2ul> j) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::cudnn_convolution_backward(vec[0], vec[1], vec[2], d_, e_, f_, g, h, i, j);
      return {std::get<0>(res), std::get<1>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1]};
}

Tensor checkpoint_cudnn_convolution_transpose(Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) {
  AT_ERROR("convolution_transpose");
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_transpose_backward(Tensor const&, Tensor const&, Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool, std::array<bool, 2ul>) {
  AT_ERROR("convolution_transpose_backward");
}

Tensor checkpoint_conv2d(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::conv2d(vec[0], vec[1], vec[2], d_, e_, f_, g)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_repeat(const Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec[0].repeat(b_)};
    };
  strongs s = {from_tensor(a), };
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm(Tensor const& a, Tensor const& b, Tensor const& c, Tensor const& d, Tensor const& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::native_batch_norm(vec[0], vec[1], vec[2], vec[3], vec[4], f, g, h);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d), from_tensor(e)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm_backward(Tensor const& a, Tensor const& b, Tensor const& c, Tensor const& d, Tensor const& e, Tensor const& f, Tensor const& g, bool h, double i, std::array<bool, 3ul> j) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::native_batch_norm_backward(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], h, i, j);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d), from_tensor(e), from_tensor(f) ,from_tensor(g)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_backward(Tensor const& a, Tensor const& b, Tensor const& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, Tensor const& g, Tensor const& h, std::array<bool, 3ul> i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::thnn_conv2d_backward(vec[0], vec[1], vec[2], d_, e_, f_, vec[3], vec[4], i);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(g), from_tensor(h)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_forward(Tensor const& a, Tensor const& b, c10::ArrayRef<long> c, Tensor const& d, c10::ArrayRef<long> e, c10::ArrayRef<long> f) {
  std::vector<long> c_ = c.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::thnn_conv2d_forward(vec[0], vec[1], c_, vec[2], e_, f_);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(d)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

Tensor checkpoint_slice(Tensor const& a, long b, long c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice(vec[0], b, c, d, e)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec[0], b_, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_slice_backward(Tensor const& a, c10::ArrayRef<long> b, long c, long d, long e, long f) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice_backward(vec[0], b_, c, d, e, f)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor, Tensor, Tensor> checkpoint_cudnn_batch_norm(Tensor const& a, Tensor const& b, Tensor const& c, Tensor const& d, Tensor const& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::cudnn_batch_norm(vec[0], vec[1], vec[2], vec[3], vec[4], f, g, h);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res), std::get<3>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d), from_tensor(e)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2], res[3]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_cudnn_batch_norm_backward(Tensor const& a, Tensor const& b, Tensor const& c, Tensor const& d, Tensor const& e, Tensor const& f, Tensor const& g, double h, Tensor const& i) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto res = at::cudnn_batch_norm_backward(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], h, vec[7]);
      return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d), from_tensor(e), from_tensor(f), from_tensor(g), from_tensor(i)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

Tensor checkpoint_select(const Tensor& a, long b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select(vec[0], b, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_ne(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec[0], b)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_ne(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec[0], vec[1])};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_embedding(const Tensor & weight, const Tensor & indices,
                            long padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::embedding(vec[0], vec[1], padding_idx, scale_grad_by_freq, sparse)};
  };
  strongs s = {from_tensor(weight), from_tensor(indices)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_embedding_backward(const Tensor & grad, const Tensor & indices, long num_weights,
                                     long padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::embedding_backward(vec[0], vec[1], num_weights, padding_idx, scale_grad_by_freq, sparse)};
  };
  strongs s = {from_tensor(grad), from_tensor(indices)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_as_strided(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::optional<long> d) {
  std::vector<long> b_ = b.vec(), c_ = c.vec();
   rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided(vec[0], b_, c_, d)};
    };
  strongs s = {from_tensor(a)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

template<typename T>
struct Ref {
  mutable T t;
};

std::tuple<Tensor, Tensor> checkpoint__fused_dropout(const Tensor & self, double p, Generator* g) {
  // TODO: Figure out how to properly duplicate the generator;
  // note that the commented-out code below results in a segfault!
  // Ref<std::shared_ptr<Generator>> gen;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    // Generator* cur = gen.t ? gen.t.get() : g;
    // auto newG = cur->clone();
    // auto res = at::_fused_dropout(vec[0], p, cur);
    // gen.t = newG;
    auto res = at::_fused_dropout(vec[0], p);
    return {std::get<0>(res), std::get<1>(res)};
  };
  strongs s = {from_tensor(self)};
  auto res = CheckPointTensorImpl::make(rt, s, false);
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_lstm(const Tensor & input, c10::ArrayRef<Tensor> hx, c10::ArrayRef<Tensor> params, bool has_biases, long num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  strongs s;
  s.push_back(from_tensor(input));
  int num_hx = 0;
  int num_params = 0;
  for (const Tensor& h : hx) {
    s.push_back(from_tensor(h));
    num_hx += 1;
  }
  for (const Tensor& p : params) {
    s.push_back(from_tensor(p));
    num_params += 1;
  }

  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::lstm(vec[0], ArrayRef<Tensor>(vec.data() + 1, num_hx),
                        ArrayRef<Tensor>(vec.data() + 1 + num_hx, num_params),
                        has_biases, num_layers, dropout,
                        train, bidirectional, batch_first);
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };

  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_lstm(const Tensor & input, const Tensor & batch_sizes,
                                                   c10::ArrayRef<Tensor> hx, c10::ArrayRef<Tensor> params,
                                                   bool has_biases, long num_layers, double dropout,
                                                   bool train, bool bidirectional) {
  strongs s;
  s.push_back(from_tensor(input));
  s.push_back(from_tensor(batch_sizes));
  int num_hx = 0;
  int num_params = 0;
  for (const Tensor& h : hx) {
    s.push_back(from_tensor(h));
    num_hx += 1;
  }
  for (const Tensor& p : params) {
    s.push_back(from_tensor(p));
    num_params += 1;
  }

  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::lstm(vec[0], vec[1],
                        ArrayRef<Tensor>(vec.data() + 2, num_hx),
                        ArrayRef<Tensor>(vec.data() + 2 + num_hx, num_params),
                        has_biases, num_layers, dropout,
                        train, bidirectional);
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };

  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

Tensor checkpoint__masked_scale(const Tensor & self, const Tensor & mask, double scale) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_masked_scale(vec[0], vec[1], scale)};
  };
  strongs s = {from_tensor(self), from_tensor(mask)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell(const Tensor& input_gates, const Tensor& hidden_gates,
                                                                    const Tensor& cx, const Tensor& input_bias,
                                                                    const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell(vec[0], vec[1], vec[2], vec[3], vec[4]);
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };
  strongs s = {from_tensor(input_gates), from_tensor(hidden_gates), from_tensor(cx),
               from_tensor_maybe(input_bias), from_tensor_maybe(hidden_bias)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell_backward(const Tensor& grad_hy, const Tensor& grad_cy, const Tensor& cx, const Tensor& cy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell_backward(vec[0], vec[1], vec[2], vec[3], vec[4], has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  strongs s = {from_tensor_maybe(grad_hy), from_tensor_maybe(grad_cy),
               from_tensor(cx), from_tensor(cy), from_tensor(workspace)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2], res[3], res[4]};
}

std::tuple<Tensor, Tensor> checkpoint__thnn_fused_gru_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell(vec[0], vec[1], vec[2], vec[3], vec[4]);
    return {std::get<0>(res), std::get<1>(res)};
  };
  strongs s = {from_tensor(input_gates), from_tensor(hidden_gates), from_tensor(hx),
               from_tensor_maybe(input_bias), from_tensor_maybe(hidden_bias)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_gru_cell_backward(const Tensor& grad_hy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell_backward(vec[0], vec[1], has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  strongs s = {from_tensor(grad_hy), from_tensor(workspace)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1], res[2], res[3], res[4]};
}

bool checkpoint_equal(const Tensor& self, const Tensor& other) {
  // should not try rematerializing this since the metadata
  // will surely be larger than a bool
  return at::equal(native::decheckpoint(self), native::decheckpoint(other));
}

Tensor checkpoint_mean(const Tensor& self, c10::optional<c10::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dtype)};
  };
  strongs s = {from_tensor(self)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

Tensor checkpoint_mean(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype) {
  std::vector<long> dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dim_, keepdim, dtype)};
  };
  strongs s = {from_tensor(self)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

std::tuple<Tensor&, Tensor&> checkpoint_max_pool2d_with_indices_out(Tensor& output, Tensor& indices, const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  std::vector<long> kernel_size_ = kernel_size.vec();
  std::vector<long> stride_ = stride.vec();
  std::vector<long> padding_ = padding.vec();
  std::vector<long> dilation_ = dilation.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor output_ = vec.at(0);
    Tensor indices_ = vec.at(1);
    at::native::max_pool2d_with_indices_out_cuda(output_, indices_, vec.at(2), kernel_size_, stride_, padding_, dilation_, ceil_mode);
  };
  CheckPointTensorImpl::mutate(mt, {output, indices, input});
  return {output, indices};
}

std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  std::vector<long> kernel_size_ = kernel_size.vec();
  std::vector<long> stride_ = stride.vec();
  std::vector<long> padding_ = padding.vec();
  std::vector<long> dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::native::max_pool2d_with_indices_cuda(vec[0], kernel_size_, stride_, padding_, dilation_, ceil_mode);
    return {std::get<0>(res), std::get<1>(res)};
  };
  strongs s = {from_tensor(self)};
  auto res = CheckPointTensorImpl::make(rt, s);
  return {res[0], res[1]};
}

Tensor& checkpoint_max_pool2d_with_indices_backward_out(Tensor& gradInput, const Tensor& gradOutput_, const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices) {
  std::vector<long> kernel_size_ = kernel_size.vec();
  std::vector<long> stride_ = stride.vec();
  std::vector<long> padding_ = padding.vec();
  std::vector<long> dilation_ = dilation.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor gradInput_ = vec.at(0);
    at::native::max_pool2d_with_indices_backward_out_cuda(gradInput_, vec.at(1), vec.at(2), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(3));
  };
  CheckPointTensorImpl::mutate(mt, {gradInput, gradOutput_, input, indices});
  return gradInput;
}

Tensor checkpoint_max_pool2d_with_indices(const Tensor& gradOutput, const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices) {
  std::vector<long> kernel_size_ = kernel_size.vec();
  std::vector<long> stride_ = stride.vec();
  std::vector<long> padding_ = padding.vec();
  std::vector<long> dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::max_pool2d_with_indices_backward_cuda(vec[0], vec[1], kernel_size_, stride_, padding_, dilation_, ceil_mode, vec[2])};
  };
  strongs s = {from_tensor(gradOutput), from_tensor(input), from_tensor(indices)};
  return CheckPointTensorImpl::make(rt, s)[0];
}

}}
