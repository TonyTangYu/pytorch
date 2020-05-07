#pragma once

#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace at {

struct CAFFE2_API CheckpointTensorCell : intrusive_ptr_target {
  Tensor t;
  explicit CheckpointTensorCell(const Tensor& t) : t(t.detach()) { }
  size_t memory() {
    return t.defined() ? t.numel() * t.itemsize() : 0;
  }
};

struct CAFFE2_API CheckpointTensorImplCell : intrusive_ptr_target {
  mutable intrusive_ptr<CheckpointTensorCell> value;
  explicit CheckpointTensorImplCell(const intrusive_ptr<CheckpointTensorCell>& value) : value(value) { }
  explicit CheckpointTensorImplCell(const Tensor& t) : value(intrusive_ptr<CheckpointTensorCell>::make(t)) { }
  void release_resources() final {
    value.reset();
  }
};

class CheckpointTensorCell;
using strong = intrusive_ptr<CheckpointTensorCell>;
using strongs = std::vector<strong>;
using weak = weak_intrusive_ptr<CheckpointTensorCell>;
using weaks = std::vector<weak>;
using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
using mutate_function_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  CHECK(!t.has(DispatchKey::CheckpointTensorId));
  auto ret = t.add(DispatchKey::CheckpointTensorId);
  CHECK(!ret.has(DispatchKey::VariableTensorId));
  return ret;
}

struct CAFFE2_API CheckpointTensorImpl : TensorImpl {
  int id = gen_counter();
  static int counter;
  static int gen_counter() {
    return counter++;
  }
  std::string counter_name() const {
    return std::string("x") + std::to_string(id);
  }
  intrusive_ptr<CheckpointTensorImplCell> ref;
  void release_resources() final;
  explicit CheckpointTensorImpl(const intrusive_ptr<CheckpointTensorImplCell>& ref) : TensorImpl(convert_key_set(ref->value->t.key_set()),
                                                                                                 ref->value->t.dtype(),
                                                                                                 ref->value->t.optional_device()), ref(ref) { }
  explicit CheckpointTensorImpl(const Tensor& t) : CheckpointTensorImpl(intrusive_ptr<CheckpointTensorImplCell>::make(t)) { }
  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      const Tensors& inputs);
  // mutate_idx indicate which of the inputs will get mutated.
  static void mutate(const std::string& name,
                     const mutate_function_t& mutate,
                     const Tensors& inputs,
                     const std::vector<size_t>& mutate_idx);
  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
  int64_t dim() const override {
    return ref->value->t.dim();
  }
  int64_t numel() const override {
    return ref->value->t.numel();
  }
  IntArrayRef sizes() const override {
    return ref->value->t.sizes();
  }
  int64_t size(int64_t d) const override {
    return ref->value->t.size(d);
  }
  IntArrayRef strides() const override {
    return ref->value->t.strides();
  }
  bool has_storage() const override {
    return false;
  }
};

inline CheckpointTensorImpl* get_cpti(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti;
}

inline Tensor get(const strong& s) {
  return s->t;
}

inline intrusive_ptr<CheckpointTensorImplCell> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

}
