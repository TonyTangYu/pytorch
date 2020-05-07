#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint_add(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("add", rt, {a, b})[0];
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("add_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_mul(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mul", rt, {a, b})[0];
}

Tensor& checkpoint_mul_(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("mul_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_abs(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  return CheckpointTensorImpl::make("abs", rt, {a})[0];
}

Tensor checkpoint_div(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("div", rt, {a, b})[0];
}

Tensor& checkpoint_div_(Tensor& a, const Tensor& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).div_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("div_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec.at(0), b_, c)};
    };
  return CheckpointTensorImpl::make("constant_pad_nd", rt, {a})[0];
}

Tensor checkpoint_binary_cross_entropy(const Tensor& a, const Tensor& b, const Tensor& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("binary_cross_entropy", rt, {a, b, c})[0];
}

Tensor& checkpoint_binary_cross_entropy_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_out(self, vec.at(1), vec.at(2), vec.at(3), e);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {a, b, c, d}, {0});
  return a;
}

Tensor checkpoint_binary_cross_entropy_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) { 
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e)};
    };
  return CheckpointTensorImpl::make("binary_cross_entropy_backward", rt, {a, b, c, d})[0];
}

Tensor& checkpoint_binary_cross_entropy_backward_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_backward_out(self, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_backward_out", mt, {a, b, c, d, e}, {0});
  return a;
}

Tensor checkpoint_embedding(const Tensor& a, const Tensor& b, long c, bool d, bool e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding(vec.at(0), vec.at(1), c, d, e)};
    };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

Tensor checkpoint_embedding_backward(const Tensor& a, const Tensor& b, long c, long d, bool e, bool f) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_backward(vec.at(0), vec.at(1), c, d, e, f)};
    };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
checkpoint_cudnn_batch_norm(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, h);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm", rt, {a, b, c, d, e});
  return {ret[0], ret[1], ret[2], ret[3]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_cudnn_batch_norm_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, at::Tensor const& d, at::Tensor const& e, at::Tensor const& f, at::Tensor const& g, double h, at::Tensor const& i) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), h, vec.at(7));
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm_backward", rt, {a, b, c, d, e, f, g, i});
  return {ret[0], ret[1], ret[2]};
}

Tensor checkpoint_as_strided(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::optional<long> d) {
  std::vector<long> b_ = b.vec(), c_ = c.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided(vec.at(0), b_, c_, d)};
    };
  return CheckpointTensorImpl::make("as_strided", rt, {a})[0];
}

Tensor checkpoint__masked_scale(const Tensor& a, const Tensor& b, double c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_masked_scale(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("_masked_scale", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution_transpose(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose(vec.at(0), vec.at(1), c_, d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, std::array<bool, 2ul> j) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g, h, i, j);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_backward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_transpose_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, long h, bool i, bool j, std::array<bool, 2ul> k) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_transpose_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g_, h, i, j, k);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_transpose_backward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

Tensor checkpoint_cudnn_convolution_backward_input(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_input(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_backward_input", rt, {b, c})[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_input(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_input(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_input", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_backward_weight", rt, {b, c})[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_weight", rt, {b, c})[0];
}

Tensor checkpoint_relu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::relu(vec.at(0))};
    };
  return CheckpointTensorImpl::make("relu", rt, {a})[0];
}

Tensor& checkpoint_relu_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).relu_();
    };
  CheckpointTensorImpl::mutate("relu_", mt, {a}, {0});
  return a;
}

std::tuple<Tensor&, Tensor&> checkpoint_max_pool2d_with_indices_out(Tensor& a, Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0), b_ = vec.at(1);
      at::max_pool2d_with_indices_out(a_, b_, vec.at(2), d_, e_, f_, g_, h);
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_out", mt, {a, b, c}, {0, 1});
  return {a, b};
}

Tensor checkpoint_avg_pool2d(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, bool e, bool f, c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d(vec.at(0), b_, c_, d_, e, f, g)};
    };
  return CheckpointTensorImpl::make("avg_pool2d", rt, {a})[0];
}

Tensor checkpoint_avg_pool2d_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d_backward(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("avg_pool2d_backward", rt, {a, b})[0];
}

Tensor& checkpoint_avg_pool2d_out(Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_out(a_, vec.at(1), c_, d_, e_, f, g, h);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_out", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_avg_pool2d_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, bool h, c10::optional<long> i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, i);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_backward_grad_input", mt, {a, b, c}, {0});
  return a;
}

std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max_pool2d_with_indices(vec.at(0), b_, c_, d_, e_, f);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("max_pool2d_backward", rt, {a});
  return {ret[0], ret[1]};
}

Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h, const Tensor& i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::max_pool2d_with_indices_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, vec.at(3));
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
  return a;
}

Tensor checkpoint_max_pool2d_with_indices_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, const Tensor& h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max_pool2d_with_indices_backward(vec.at(0), vec.at(1), c_, d_, e_, f_, g, vec.at(2))};
    };
  return CheckpointTensorImpl::make("max_pool2d_with_indices_backward", rt, {a, b, h})[0];
}

Tensor checkpoint_view(const Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).view(b_)};
    };
  return CheckpointTensorImpl::make("view", rt, {a})[0];
}

Tensor checkpoint_ne_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("ne_Scalar", rt, {a})[0];
}

Tensor& checkpoint_ne_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("ne_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_ne_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("ne_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_ne_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("ne_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_eq_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("eq_Scalar", rt, {a})[0];
}

Tensor& checkpoint_eq_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("eq_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_eq_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("eq_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_eq_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("eq_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_addmm(const Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addmm(vec.at(0), vec.at(1), vec.at(2), d, e)};
    };
  return CheckpointTensorImpl::make("addmm", rt, {a, b, c})[0];
}

Tensor& checkpoint_addmm_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, c10::Scalar e, c10::Scalar f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::addmm_out(a_, vec.at(1), vec.at(2), d, e, f);
    };
  CheckpointTensorImpl::mutate("addmm_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_addmm_(Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.addmm_(vec.at(1), vec.at(2), d, e);
    };
  CheckpointTensorImpl::mutate("addmm_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_sigmoid(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sigmoid", rt, {a})[0];
}

Tensor& checkpoint_sigmoid_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.sigmoid_();
    };
  CheckpointTensorImpl::mutate("sigmoid_", mt, {a}, {0});
  return a;
}

Tensor checkpoint__log_softmax(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("_log_softmax", rt, {a})[0];
}

Tensor checkpoint__log_softmax_backward_data(const Tensor& a, const Tensor& b, long c, const Tensor& d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax_backward_data(vec.at(0), vec.at(1), c, vec.at(2))};
    };
  return CheckpointTensorImpl::make("_log_softmax_backward_data", rt, {a, b, d})[0];
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss_forward(const Tensor& a, const Tensor& b, const Tensor& c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::nll_loss_forward(vec.at(0), vec.at(1), vec.at(2), d, e);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("nll_loss_forward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

std::tuple<Tensor&, Tensor&> checkpoint_nll_loss_forward_out(Tensor& a, Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      Tensor b_ = vec.at(1);
      at::nll_loss_forward_out(a_, b_, vec.at(2), vec.at(3), vec.at(4), f, g);
    };
  CheckpointTensorImpl::mutate("nll_loss_forward_out", mt, {a, b, c, d, e}, {0, 1});
  return {a, b};
}

Tensor checkpoint_nll_loss_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e, long f, const Tensor& g) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nll_loss_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e, f, vec.at(4))};
    };
  return CheckpointTensorImpl::make("nll_loss_backward", rt, {a, b, c, d, g})[0];
}

Tensor& checkpoint_nll_loss_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g, const Tensor& h) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::nll_loss_backward_out(a_, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, vec.at(5));
    };
  CheckpointTensorImpl::mutate("nll_loss_backward_grad_input", mt, {a, b, c, d, e, h}, {0});
  return a;
}

Tensor checkpoint_mm(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mm(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mm", rt, {a, b})[0];
}

Tensor& checkpoint_mm_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::mm_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("mm_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_sum(const Tensor& a, c10::optional<c10::ScalarType> b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("sum", rt, {a})[0];
}

Tensor checkpoint_sum_dim_IntList(const Tensor& a, c10::ArrayRef<long> b, bool c, c10::optional<c10::ScalarType> d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("sum_dim_IntList", rt, {a})[0];
}

Tensor checkpoint_threshold(const Tensor& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("threshold", rt, {a})[0];
}

Tensor& checkpoint_threshold_(Tensor& a, c10::Scalar b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_(a_, b, c);
    };
  CheckpointTensorImpl::mutate("threshold_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_threshold_out(Tensor& a, const Tensor& b, c10::Scalar c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_out(a_, b, c, d);
    };
  CheckpointTensorImpl::mutate("threshold_out", mt, {a}, {0});
  return a;
}

Tensor checkpoint_threshold_backward(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold_backward(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("threshold_backward", rt, {a, b})[0];
}

Tensor checkpoint_select(const Tensor& a, long b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("select", rt, {a})[0];
}

Tensor checkpoint_select_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select_backward(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("select_backward", rt, {a})[0];
}

Tensor checkpoint_slice(const Tensor& a, long b, long c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice(vec.at(0), b, c, d, e)};
    };
  return CheckpointTensorImpl::make("slice", rt, {a})[0];
}

Tensor checkpoint_slice_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d, long e, long f) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice_backward(vec.at(0), b_, c, d, e, f)};
    };
  return CheckpointTensorImpl::make("slice_backward", rt, {a})[0];
}

Tensor& checkpoint_zero_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).zero_();
    };
  CheckpointTensorImpl::mutate("zero_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a, at::Dimname b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_();
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a, long b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_sigmoid_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("sigmoid_backward", rt, {a, b})[0];
}

Tensor& checkpoint_sigmoid_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sigmoid_backward_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("sigmoid_backward_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_sign_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sign_out(a_, vec.at(1));
    };
  CheckpointTensorImpl::mutate("sign_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_sign(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sign(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sign", rt, {a})[0];
}

Tensor checkpoint_tanh(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh(vec.at(0))};
    };
  return CheckpointTensorImpl::make("tanh", rt, {a})[0];
}

Tensor checkpoint_tanh_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("tanh_backward", rt, {a, b})[0];
}

Tensor& checkpoint_tanh_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::tanh_backward_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("tanh_backward_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_neg(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::neg(vec.at(0))};
    };
  return CheckpointTensorImpl::make("neg", rt, {a})[0];
}

Tensor checkpoint_sub(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("sub", rt, {a, b})[0];
}

Tensor& checkpoint_sub_(at::Tensor& a, at::Tensor const& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    self.sub_(vec.at(1), c);
  };
  CheckpointTensorImpl::mutate("sub_", mt, {a, b}, {0});
  return a;
}


Tensor checkpoint_repeat(const at::Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).repeat(b_)};
    };
  return CheckpointTensorImpl::make("repeat", rt, {a})[0];
}

Tensor checkpoint_mean(const Tensor& self, c10::optional<c10::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dtype)};
  };
  return CheckpointTensorImpl::make("mean", rt, {self})[0];
}

Tensor checkpoint_mean(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype) {
  std::vector<long> dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::native::mean_cpu_gpu(vec[0], dim_, keepdim, dtype)};
  };
  return CheckpointTensorImpl::make("mean.dim", rt, {self})[0];
}

Tensor checkpoint__cat(c10::ArrayRef<Tensor> a, long b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cat(vec, b)};
    };
  std::vector<Tensor> s;
  for (const Tensor& t : a) {
    s.push_back(t);
  }
  return CheckpointTensorImpl::make("_cat", rt, s)[0];
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
  CheckpointTensorImpl::mutate("_cat_out", mt, args, {0});
  return a;
}

Tensor checkpoint_kl_div(at::Tensor const& a, at::Tensor const& b, long c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("kl_div", rt, {a, b})[0];
}

Tensor checkpoint_kl_div_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div_backward(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("kl_div_backward", rt, {a, b, c})[0];
}

Tensor checkpoint_upsample_bilinear2d(at::Tensor const& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d(vec.at(0), output_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d", rt, {self})[0];
}

Tensor& checkpoint_upsample_bilinear2d_out(at::Tensor& out, const at::Tensor& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out = vec.at(0);
    at::upsample_bilinear2d_out(out, vec.at(1), output_size_, align_corners, scales_h, scales_w);
  };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {out, self}, {0});
  return out;
}

Tensor& checkpoint_upsample_bilinear2d_backward_out(at::Tensor& grad_input, const at::Tensor& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input = vec.at(0);
    at::upsample_bilinear2d_backward_out(grad_input, vec.at(1), output_size_, input_size_, align_corners, scales_h, scales_w);
  };
  CheckpointTensorImpl::mutate("upsample_bilinear2d_backward_out", mt, {grad_input, grad_output}, {0});
  return grad_input;
}

Tensor checkpoint_upsample_bilinear2d_backward(at::Tensor const& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d_backward(vec.at(0), output_size_, input_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d_backward", rt, {grad_output})[0];
}

Tensor& checkpoint_clamp_min_(Tensor& a, Scalar min) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::clamp_min_(self, min);
  };
  CheckpointTensorImpl::mutate("clamp_min_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_clamp_min__out(Tensor& out, const Tensor& self, Scalar min) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out = vec.at(0);
    at::clamp_min_out(out, vec.at(1), min);
  };
  CheckpointTensorImpl::mutate("clamp_min__out", mt, {out, self}, {0});
  return out;
}

Tensor checkpoint_binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::binary_cross_entropy_with_logits(vec.at(0), vec.at(1), vec.at(2), vec.at(3), reduction)};
  };
  return CheckpointTensorImpl::make("binary_cross_entropy_with_logits", rt, {input, target, weight, pos_weight})[0];
}

Tensor checkpoint_binary_cross_entropy_with_logits_backward(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::binary_cross_entropy_with_logits_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), reduction)};
  };
  return CheckpointTensorImpl::make("binary_cross_entropy_with_logits_backward", rt, {grad, input, target, weight, pos_weight})[0];
}

std::tuple<Tensor, Tensor> checkpoint__fused_dropout(const Tensor & self, double p, Generator* g) {
  // TODO: Figure out how to properly duplicate the generator;
  // note that the commented-out code below results in a segfault!
  // Ref<std::shared_ptr<Generator>> gen;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    // Generator* cur = gen.t ? gen.t.get() : g;
    // auto newG = cur->clone();
    // auto res = at::_fused_dropout(vec.at(0), p, cur);
    // gen.t = newG;
    auto res = at::_fused_dropout(vec.at(0), p);
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_fused_droupout_", rt, {self});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& cx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell(vec.at(0), vec.at(1), vec.at(2),
                                         vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell", rt,
                                        {input_gates, hidden_gates, cx, input_bias, hidden_bias});
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell_backward(const Tensor& grad_hy, const Tensor& grad_cy, const Tensor& cx, const Tensor& cy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell_backward", rt,
                                        {grad_hy, grad_cy, cx, cy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

std::tuple<Tensor, Tensor> checkpoint__thnn_fused_gru_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell", rt,
                                        {input_gates, hidden_gates, hx, input_bias, hidden_bias});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_gru_cell_backward(const Tensor& grad_hy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell_backward(vec.at(0), vec.at(1), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell_backward", rt,
                                        {grad_hy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

Scalar checkpoint__local_scalar_dense(at::Tensor const& a) {
  return at::_local_scalar_dense(decheckpoint(a));
}

}}
