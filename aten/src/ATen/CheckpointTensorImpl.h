#pragma once

#include <random>

#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/Logger.h>

// [System Description]:
// Every Tensor is managed by a CheckpointTensor,
// that describe how it is computed, (the function and the inputs)
// And might optionally hold the tensor value.
// The tensor value might be dropped, and when requested later, recomputed and cached again.

// [DTR and autodiff]:
// unlike other gradient checkpointing scheme that is coupled to automatic differentation,
// DTR is decoupled from automatic checkpointing, and thus is more like a tensor level cache
// then memory optimization on automatic differentiation.
// this mean we can use DTR without using autodiff.
// Implementation wise, this mean DTR will work below autodiff:
// let autodiff do it's thing, and DTR will override the forward and backward propagation tensor.

// [Corner Cases]:
// A CheckpointedTensor might require_grad.
//   In this case the underlying data must not require_grad,
//   as we want backpropagation on the outer, uncheckpoined level.
//   see Note [DTR and autodiff].
// A CheckpointedTensor might be constant.
//   In this case it is unevictable.
// An operator might return multiple output.
//   In this case the computation info (rematerializer) is shared between all of them,
//   And when the function get computed again all value get cached.
// An operator might mutate input value.
//   To combat this, we COW the operator, and wrap CheckpopintTensor with a Ref.
//   By doing this the inner CheckpointTensor is kept purely functional.
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
// When all external reference to a Checkpoint Tensor is lost,
// we will try to immediately evict it.

// Note: to code fast I do not use RAII and just assume the code will not try to recover from exception.
// It should be easy to fix though.

namespace at {

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Duration = Clock::duration;

// TODO: using a pool allocator might make more sense - no need to allocate and delete each pointer individually.
// TODO: egg simply store all node in a vector and use vector index as reference. maybe do that?
template<typename T>
struct EquivalentClassNode : intrusive_ptr_target {
  explicit EquivalentClassNode(const T& t) : t_unsafe(t) { }
  mutable intrusive_ptr<EquivalentClassNode> parent;
  bool is_root() {
    return !parent;
  }
  void release_resources() override {
    parent.reset();
  }
  T t_unsafe;
};

template<typename T>
T& get_t(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  return find_root(n)->t_unsafe;
}

template<typename T>
static void update_t(const intrusive_ptr<EquivalentClassNode<T>>& n, const T& t) {
  find_root(n)->t_unsafe = t;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> find_root(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  if (n->is_root()) {
    return n;
  } else {
    n->parent = find_root(n->parent);
    return n->parent;
  }
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> merge(const std::function<T(const T&, const T&)>& merge_t,
                                            const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  if (l == r) {
    return l;
  }
  l->parent = r;
  r->t_unsafe = merge_t(l->t_unsafe, r->t_unsafe);
  return r;
}

size_t memory(const Tensor& t);

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
// do we really need this? can we just use operatorhandle?
// after some thoght i think we shouldnt be that coupled.
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
// this doesnt look right. a mutate function can also return input.
// i guess mutate_function simply is not as fundamental as rematerialize_function.
// the former desugar to the latter.
// using mutate_function_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;
struct CheckpointInfo {
  duration_t compute_cost;
  double cost(size_t memory, size_t staleness) const {
    TORCH_CHECK(memory > 0);
    TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  CheckpointInfo(duration_t compute_cost) :
    compute_cost(compute_cost) {
  }
};

// ecn represent a evicted tensor group.
// it is a set of tensor that are evicted, and if two evicted tensor are input -> output to each other,
// they must be in an ecn.
// note: we try to support removal from ecn by subtracting compute_cost and memory.
// this will create suprious connection but that should be fine empircally.
// below is an example of a suprious connection:
// a -> b, a -> c
// a, b, c got evicted so belong to a single ecn.
// a got rematerialized.
// b, c still belong to a single ecn although there is no connection.
using ecn_ptr = intrusive_ptr<EquivalentClassNode<CheckpointInfo>>;

struct Unsafe { };

// The rematerializer could be called to reinvoke an operator.
// Tensor point to remat which point to Tensor.
// To build the cycle remat support a default constructor,
// And allow you to fill in the member later.
struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  strongs inputs;
  weaks outputs;
  duration_t compute_cost;
  // when some output in here get evicted, they should belong to this ecn.
  // a rematerializer have to track this,
  // because when multiple output of a rematerializer get evicted,
  // we only want to count the compute cost once.
  ecn_ptr ecn;
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
                 duration_t compute_cost)  :
    func(func),
    inputs(inputs),
    compute_cost(compute_cost) {
  }
  void release_resources() final {
    func = rematerialize_function_t();
    inputs.clear();
    outputs.clear();
  }
  void remat();
  ecn_ptr get_ecn();
};

// Track all Tensor that share the same Storage.
// This is the atomic level of eviction - when evicting, everything here will get evicted.
// When an AliasPool is evicted, the Storage of the underlying tensor must be freed.
// Additionally, the AliasPool contain weak pointer to all children of tensors,
// in order to compute the score of evicting a Storage.
struct AliasPool : intrusive_ptr_target {
  weaks tensors;
  weaks neighbors;
  std::set<ecn_ptr> neighbor_ecn();
  // get() might hold some raw Tensor, rendering them unevictable.
  // it is likely that get() will run out of memory, and when it does so, it will try to evict.
  // so, it is crucial that we dont try to evict those tensors - doing so will not evict anything.
  // lock_count count how many time a tensor is referenced by get.
  size_t lock_count = 0;
  size_t external_count = 0;
  void lock() {
    ++lock_count;
  }
  void unlock() {
    TORCH_CHECK(lock_count > 0);
    --lock_count;
  }
  intrusive_ptr<Rematerializer> head_remat;
  size_t memory;
  time_t last_used_time;
  // An aliaspool cant register itself to the checkpointpool - you have to do it yourself.
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory) :
    head_remat(head_remat),
    memory(memory),
    last_used_time(std::chrono::system_clock::now()) {
  }
  // hold the evicted tensor group if it is evicted.
  // is empty if it is not evicted
  ecn_ptr ecn;
  double cost(time_t current_time);
  bool evictable() const {
    return lock_count == 0 && head_remat && !ecn;
  }
  void evict();
  void register_external() {
    ++external_count;
  }
  void release_external() {
    TORCH_CHECK(external_count > 0);
    --external_count;
    if (external_count == 0 && evictable()) {
      evict();
    }
  }
  // if it was evicted, refresh it. otherwise do nothing.
  // have to check so, because when we rematerialize a single tensor in an aliaspool,
  // we will set it to non-evicted, and when we rematerialize it's tensor they will also reset this.
  void set_not_evicted(const intrusive_ptr<AliasPool>& self);
  void release_resources() final {
    tensors.clear();
    neighbors.clear();
    head_remat.reset();
  }
};

struct CheckpointTensorCell : intrusive_ptr_target {
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
  // A Tensor is evictable iff it's AliasPool is evictable.
  // A evictable tensor must have Rematerializer.
  intrusive_ptr<AliasPool> pool;
  intrusive_ptr<Rematerializer> remat;
  void evict() {
    TORCH_CHECK(remat);
    t.reset();
  }
  void fill(const Tensor& t);
  explicit CheckpointTensorCell(const Tensor& t, const intrusive_ptr<AliasPool>& pool) : pool(pool) {
    fill(t);
  }
  explicit CheckpointTensorCell(const Tensor& t,
                                const intrusive_ptr<AliasPool>& pool,
                                const intrusive_ptr<Rematerializer>& remat) :
    pool(pool), remat(remat) {
    fill(t);
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
    TORCH_CHECK(! t->key_set().has(DispatchKey::Checkpoint));
    TORCH_CHECK(! t->key_set().has(DispatchKey::Autograd));
    pool->last_used_time = std::chrono::system_clock::now();
    return *t;
  }
  // pin() make a cell unevictable.
  // This allow us to deallocate the rematerializer,
  // and potentially tensors that are only used by the rematerializer.
  // todo: pin() in the paper mean lock(). find better name?
  void pin() {
    get();
    pool->head_remat.reset();
    remat.reset();
  }
  void release_resources() final {
    t.reset();
    pool.reset();
    remat.reset();
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
  External(const strong& value) : value(value) {
    value->pool->register_external();
  }
  External(const Tensor& value) :
    External(strong::make(value,
                          intrusive_ptr<AliasPool>::make(Unsafe(),
                                                         intrusive_ptr<Rematerializer>(),
                                                         memory(value)))) { }
  External(const Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    External(strong::make(value, pool, remat)) { }
  strong value;
  void release_resources() override;
};

struct TORCH_API CheckpointTensorImpl : TensorImpl {
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

  explicit CheckpointTensorImpl(const Ref<intrusive_ptr<External>>& ref);

  explicit CheckpointTensorImpl(const intrusive_ptr<External>& e) :
    CheckpointTensorImpl(Ref<intrusive_ptr<External>>::make(e)) { }

  explicit CheckpointTensorImpl(const Tensor& t);

  Tensor get() const;

  static Tensors make(const std::string& name,
                      const rematerialize_function_t& remat,
                      const Tensors& inputs);

  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;

  intrusive_ptr<TensorImpl> shallow_copy_and_detach(VariableVersion&& version_counter,
                                                    bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  int64_t dim() const override {
    return get().dim();
  }
  int64_t numel() const override {
    return get().numel();
  }
  IntArrayRef sizes() const override {
    return get().sizes();
  }
  int64_t size(int64_t d) const override {
    return get().size(d);
  }
  IntArrayRef strides() const override {
    return get().strides();
  }
  int64_t stride(int64_t d) const override {
    return get().stride(d);
  }
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    return get().is_contiguous(memory_format);
  }
  bool has_storage() const override {
    return false;
  }
};

// CheckpointPool manage all the CheckpointTensor.
// It allow one to:
// 0: Search over all aliaspool to evict tensors.
// 1: pin all the tensors.
struct CheckpointPool {
  std::vector<weak_intrusive_ptr<AliasPool>> aps;
  std::vector<weak_intrusive_ptr<External>> exts;
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  // whether to take a square-root sample of the pool during an eviction loop
  bool sample_tensors = true;
  // ignore tensors < 1% of the average tensor size
  bool ignore_small_tensors = true;
  bool has_memory_budget = false;
  long memory_budget;
  void evict();
  void auto_evict();
  void clear_checkpointpool();
  void add(const intrusive_ptr<AliasPool>&);
};

struct CheckpointFunctions;
TORCH_API CheckpointFunctions* GetCheckpointFunctions();

// using function pointer to pass through linking boundary
struct CheckpointFunctions {
  virtual ~CheckpointFunctions() { }
#define DefineCheckpointFunction(RETURN, NAME, ...)     \
  virtual RETURN NAME(__VA_ARGS__) = 0;                 \
  static RETURN static_ ## NAME(__VA_ARGS__) {          \
    return GetCheckpointFunctions()->NAME(__VA_ARGS__); \
  }
  DefineCheckpointFunction(void, new_log, std::string(str));
  DefineCheckpointFunction(void, annotate_log, std::string(str));
  DefineCheckpointFunction(void, toggle_log, bool(log));
  DefineCheckpointFunction(void, clear_checkpointpool);
  DefineCheckpointFunction(void, unset_memory_budget);
  DefineCheckpointFunction(void, set_memory_budget, long(budget));
  DefineCheckpointFunction(void, toggle_sampling, bool(sample));
  DefineCheckpointFunction(void, toggle_ignore_small_tensors, bool(ignore));
  DefineCheckpointFunction(void, reset_profile);
  DefineCheckpointFunction(void, toggle_profile, bool(profile));
  DefineCheckpointFunction(long, base_compute_time);
  DefineCheckpointFunction(long, remat_compute_time);
  DefineCheckpointFunction(long, compute_time);
  DefineCheckpointFunction(long, cost_time);
  DefineCheckpointFunction(long, search_time);
  DefineCheckpointFunction(long, loop_time);
};

}
