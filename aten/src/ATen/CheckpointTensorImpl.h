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

void DTRLog(const std::string& str);

struct CAFFE2_API CheckpointTensorCell : intrusive_ptr_target {
  Tensor t;
  explicit CheckpointTensorCell(const Tensor& t) : t(t.detach()) { }
  int id = gen_counter();
  static int counter;
  static int gen_counter() {
    return counter++;
  }
  std::string name() {
    return std::string("x") + std::to_string(id);
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

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  auto ret = t.add(DispatchKey::CheckpointTensorId);
  CHECK(!ret.has(DispatchKey::VariableTensorId));
  return ret;
}

struct CAFFE2_API CheckpointTensorImpl : TensorImpl {
  intrusive_ptr<CheckpointTensorImplCell> ref;
  void release_resources() final {
    ref.reset();
  }
  explicit CheckpointTensorImpl(const intrusive_ptr<CheckpointTensorImplCell>& ref) : TensorImpl(convert_key_set(ref->value->t.key_set()),
                                                                                                 ref->value->t.dtype(),
                                                                                                 ref->value->t.optional_device()), ref(ref) { }
  explicit CheckpointTensorImpl(const Tensor& t) : CheckpointTensorImpl(intrusive_ptr<CheckpointTensorImplCell>::make(t)) { }
  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      const strongs& input_values);
  static void mutate(const std::string& name,
                     const mutate_function_t& mutate,
                     const Tensors& input_values);
};

inline CheckpointTensorImpl* get_cpti(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti;
}

inline strong from_tensor(const Tensor& t) {
  auto* cpt = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpt != nullptr) {
    return cpt->ref->value;
  } else {
    return get_cpti(native::checkpoint(t))->ref->value;
  }
}

inline Tensor get(const strong& s) {
  return s->t;
}

inline intrusive_ptr<CheckpointTensorImplCell> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

}
