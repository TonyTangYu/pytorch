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

// System Description:
// Every Tensor is managed by a CheckPointTensor,
// that describe how it is computed, (the function and the inputs)
// And might optionally hold the tensor value.
// The tensor value might be dropped, and when requested later, recomputed and cached again.

// Corner Cases:
// An CheckPointedTensor might be constant.
//   In this case it is unevictable.
// An input might be uncheckpointed.
//   In this case it is treated as a small constant and omitted from the system - it will be unevictable.
// An operator might return multiple output.
//   In this case the computation info (rematerializer) is shared between all of them,
//   And when the function get computed again all value get cached.
// An operator might not return value, but only mutate input value.
//   To combat this, we COW the operator, and wrap CheckPopintTensor with a Ref.
//   By doing this the inner CheckPointTensor is kept purely functional.
// An operator might try to mutate uncheckpointed tensor.
//   We do not support this and will error.
// An operator might create aliases.
//   We track alias in AliasPool.
//   Each AliasPool hold a set of tensor that is alias to eachother.
// An operator might try to create Alias to an unevictable tensor.
//   In such a case the output tensor is unevictable.
// An operator might try to mutate Tensor with Alias.
//   We do not support this case an will error if a Tensor has any alive Alias.
//   However it could be done without a major redesign of the system -
//   Each AliasPool will hold weak pointers to the External Reference.
//   When alias mutation occur,
//   we make a rematerialize_function that take in the base tensor (other tensor alias from)
//   and output all the new value of the aliases, then update the Ref.
//   Of course, the cleaner way is to not support this.
//   Shame on those who use this feature.

// Memory Safety:
// The objects here will have lots of backedges.
// In order to collect memory when computation is completed,
// We require that all strong pointer is of the form of value -> input.
// This ensure that everything will be released if there is no external ref whatsoever.

// Optimization:
// We treat tensor that has no external reference differently -
// They will never be externally used again so we assume their next use time is infinite
// so, if it doesnt has any evicted neighbor it will get evicted immediately.

// Note: to code fast I do not use RAII and just assume the code will not try to recover from exception.
// It should be easy to fix though.

namespace at {

inline size_t memory(const Tensor& t) {
  if (! t.has_storage()) {
    return 0;
  }
  auto& storage = t.storage();
  return storage.numel() * storage.itemsize();
}

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;
  void release_resources() final {
    static_release_resources(value);
  }
  RefCell(const T& t) : value(t) { }
};

template<typename T>
using Ref = intrusive_ptr<RefCell<T>>;

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) {
  ptr.reset();
}

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

struct Unsafe { };

// Track all Tensor that share the same Storage.
// This is the atomic level of eviction - when evicting, everything here will get evicted.
// When an AliasPool is evicted, the Storage of the underlying tensor must be freed.
// Additionally, the AliasPool contain weak pointer to all children of tensors,
// in order to compute the score of evicting a Storage.
struct AliasPool : intrusive_ptr_target {
  weaks tensors;
  // get() might hold some raw Tensor, rendering them unevictable.
  // it is likely that get() will run out of memory, and when it does so, it will try to evict.
  // so, it is crucial that we dont try to evict those tensors - doing so will not evict anything.
  // lock_count count how many time a tensor is referenced by get.
  size_t lock_count;
  bool evictable;
  size_t memory;
  AliasPool(const Unsafe&, bool evictable, size_t memory) :
    lock_count(0), evictable(evictable), memory(memory) {
  }
  void evict();
  void release_resources() final {
    tensors.clear();
  }
};

// The rematerializer could be called to reinvoke an operator.
// Tensor point to remat which point to Tensor.
// To build the cycle remat support a default constructor,
// And allow you to fill in the member later.
struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  strongs inputs;
  weaks outputs;
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs)  :
    func(func), inputs(inputs) {
  }
  void release_resources() final {
    func = rematerialize_function_t();
    inputs.clear();
    outputs.clear();
  }
  void remat();
};

struct CAFFE2_API CheckpointTensorCell : intrusive_ptr_target {
  std::unique_ptr<Tensor> t;
  bool defined = false;
  bool is_undefined_tensor;
  DispatchKeySet key_set_;
  DispatchKeySet key_set() const {
    TORCH_CHECK(defined);
    return key_set_;
  }
  caffe2::TypeMeta dtype_;
  caffe2::TypeMeta dtype() const {
    TORCH_CHECK(defined);
    return dtype_;
  }
  c10::optional<Device> optional_device_;
  c10::optional<Device> optional_device() const {
    TORCH_CHECK(defined);
    return optional_device_;
  }
  int64_t dim_, numel_;
  size_t itemsize_;
  std::vector<long> sizes_, strides_;
  // A Tensor is evictable iff it's AliasPool is evictable.
  // A evictable tensor must have Rematerializer.
  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;
  int64_t dim() const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return dim_;
  }
  int64_t numel() const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return numel_;
  }
  IntArrayRef sizes() const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return sizes_;
  }
  int64_t size(int64_t d) const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return sizes_[d];
  }
  IntArrayRef strides() const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return strides_;
  }
  int64_t stride(int64_t d) const {
    TORCH_CHECK(defined && !is_undefined_tensor);
    return strides_[d];
  }
  void evict() {
    t.reset();
  }
  void fill(const Tensor& t) {
    if (!(this->t)) {
      this->t = std::make_unique<Tensor>(t.detach());
      if (!defined) {
        defined = true;
        is_undefined_tensor = !t.defined();
        key_set_ = t.key_set();
        dtype_ = t.dtype();
        optional_device_ = t.optional_device();
        if (! is_undefined_tensor) {
          dim_ = t.dim();
          numel_ = t.numel();
          itemsize_ = t.itemsize();
          sizes_ = t.sizes().vec();
          strides_ = t.strides().vec();
        }
      }
    }
  }
  explicit CheckpointTensorCell(const Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
    fill(t);
  }
  explicit CheckpointTensorCell(const Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat) :
    pool(pool), remat(remat) {
    fill(t);
  }
  size_t itemsize() {
    return itemsize_;
  }
  size_t memory() {
    TORCH_CHECK(defined);
    return pool->memory;
  }
  Tensor get() {
    if (! t) {
      TORCH_CHECK(remat);
      remat->remat();
    }
    TORCH_CHECK(t);
    TORCH_CHECK(! t->key_set().has(DispatchKey::CheckpointTensorId))
    return *t;
  }
  void pin() {
    pool->evictable = false;
    get();
    remat.reset();
  }
  void release_resources() final {
    t.reset();
    pool.reset();
    remat.reset();
  }
};

// CheckpointPool keep a list of AliasPool, and search over them to choose the best one to evict.
struct CheckpointPool {
  static CheckpointPool& singleton() {
    static CheckpointPool cpp;
    return cpp;
  }
};

// An external reference.
// Each strong will have at most one external reference.
// By keeping such an invariant, whenever an external reference die,
// We know that the underlying strong is only used internally.
// Thus, when it die we can apply optimization like banishing/infinite staleness.
// We keep this invariant by only allowing CheckpointTensorImpl to make new External,
// When new CheckpointTensorImpl is constructed.
struct External : intrusive_ptr_target {
  External(const strong& value) : value(value) { }
  External(const Tensor& value) :
    value(intrusive_ptr<CheckpointTensorCell>::make(value,
                                                    intrusive_ptr<AliasPool>::make(Unsafe(),
                                                                                   false,
                                                                                   memory(value)))) { }
  External(const Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    value(intrusive_ptr<CheckpointTensorCell>::make(value, pool, remat)) { }
  strong value;
  void release_resources() override;
};

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
  Ref<intrusive_ptr<External>> ref;
  void release_resources() final;
  explicit CheckpointTensorImpl(const Ref<intrusive_ptr<External>>& ref) :
    TensorImpl(convert_key_set(ref->value->value->key_set()),
               ref->value->value->dtype(),
               ref->value->value->optional_device()),
    ref(ref) { }
  explicit CheckpointTensorImpl(const intrusive_ptr<External>& e) :
    CheckpointTensorImpl(Ref<intrusive_ptr<External>>::make(e)) { }
  explicit CheckpointTensorImpl(const Tensor& t) :
    CheckpointTensorImpl(intrusive_ptr<External>::make(t)) { }
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
    return ref->value->value->dim();
  }
  int64_t numel() const override {
    return ref->value->value->numel();
  }
  IntArrayRef sizes() const override {
    return ref->value->value->sizes();
  }
  int64_t size(int64_t d) const override {
    return ref->value->value->size(d);
  }
  IntArrayRef strides() const override {
    return ref->value->value->strides();
  }
  int64_t stride(int64_t d) const override {
    return ref->value->value->stride(d);
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

inline Ref<intrusive_ptr<External>> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

}
