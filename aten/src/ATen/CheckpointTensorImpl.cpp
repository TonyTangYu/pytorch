#include <ATen/CheckpointTensorImpl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at {

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Duration = Clock::duration;

DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  TORCH_CHECK(!t.has(DispatchKey::Checkpoint));
  auto ret = t.add(DispatchKey::Checkpoint);
  return ret;
}

void External::release_resources() {
  value->pool->release_external();
  value.reset();
}

CheckpointTensorImpl* get_cpti(const Tensor& t) {
  return dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
}

CheckpointTensorImpl* must_get_cpti(const Tensor& t) {
  auto ret = get_cpti(t);
  TORCH_CHECK(ret);
  return ret;
}

size_t memory_sum = 0;
size_t memory_max = 0;
size_t memory_count = 0;

void reset_memory_stat() {
  memory_sum = 0;
  memory_max = 0;
  memory_count = 0;
}

// todo: use defensive programming to make this only pass on dense tensor.
// todo: rn this track memory from all device. but if we are cpointing on gpu, we dont care about cpu.
size_t memory(const Tensor& t) {
  if (! t.has_storage()) {
    return 0;
  }
  auto& storage = t.storage();
  size_t res = storage.nbytes();
  memory_sum += res;
  memory_max = std::max(memory_max, res);
  memory_count += 1;
  return res;
}

// todo: generalize this to other device? e.g. we might want checkpointing on pure cpu.
long current_memory() {
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  return device_stat.allocated_bytes[0].current;
}

void CheckpointPool::auto_evict() {
  if (has_memory_budget) {
    while (current_memory() > memory_budget) {
      evict();
    }
  }
}

bool use_log_ = false;
bool use_profile_ = false;
long base_compute_time_ = 0;
long remat_compute_time_ = 0;
long search_time_ = 0;
long cost_time_ = 0;

CheckpointPool pool;
void CheckpointPool::add(const intrusive_ptr<AliasPool>& p) {
  if (p->memory > 0 && (memory_count == 0 || !ignore_small_tensors || p->memory >= 0.01 * double(memory_sum/memory_count))) {
    aps.push_back(weak_intrusive_ptr<AliasPool>(p));
  }
}

CheckpointTensorImpl::CheckpointTensorImpl(const Tensor& t) : CheckpointTensorImpl(intrusive_ptr<External>::make(t)) { }

CheckpointTensorImpl::CheckpointTensorImpl(const Ref<intrusive_ptr<External>>& ref) :
  TensorImpl(convert_key_set(ref->value->value->key_set()),
             ref->value->value->dtype(),
             ref->value->value->optional_device()),
  ref(ref) {
  if (key_set().has(DispatchKey::Autograd)) {
    set_requires_grad(true);
  }
}

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  auto ret = intrusive_ptr<CheckpointTensorImpl>::make(ref);
  if (use_log_) {
    DTRLogCopy(ret->counter_name(), counter_name());
  }
  return ret;
}

void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(impl->key_set().has(DispatchKey::Checkpoint));
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(impl.get());
  TORCH_CHECK(cpti != nullptr);
  ref->value = cpti->ref->value;
  if (use_log_) {
    DTRLogCopyFrom(counter_name(), cpti->counter_name());
  }
}

int CheckpointTensorImpl::counter = 0;

bool is_alias(const Tensor& l, const Tensor& r) {
  return l.defined() && r.defined() && l.is_alias_of(r);
}

// return an index for alias.
// we dont care which one because they all lead to the same alias pool.
// return -1 for no alias.
// may god forgive my sin.
int get_alias(const Tensors& ts, const Tensor& t) {
  if (t.defined()) {
    for (size_t i = 0; i < ts.size(); ++i) {
      if (ts[i].defined() && t.is_alias_of(ts[i])) {
        return i;
      }
    }
  }
  return -1;
}

void CheckpointTensorImpl::release_resources() {
  if (use_log_) {
    DTRLogRelease(counter_name());
  }
  ref.reset();
}

namespace native {

Tensor checkpoint(const Tensor& t) {
  TORCH_CHECK(!is_checkpoint(t));
  auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t);
  if (use_log_) {
    DTRLogConstant(cpti->counter_name());
    DTRLogMemory(cpti->counter_name(), cpti->ref->value->value->memory());
  }
  return Tensor(cpti);
}

Tensor uncheckpoint(const Tensor& t) {
  auto cpti = must_get_cpti(t);
  return cpti->get();
}

Tensor try_uncheckpoint(const Tensor& t) {
  return is_checkpoint(t) ? uncheckpoint(t) : t;
}

Tensor decheckpoint(const Tensor& t) {
  return try_uncheckpoint(t);
}

void pin(Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  cpti->ref->value->value->pin();
}

bool is_checkpoint(const Tensor& t) {
  return get_cpti(t) != nullptr;
}

Tensor try_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? t : checkpoint(t);
}

void new_log(std::string str) {
  DTRLogger::logger().out = std::ofstream(DTRLogger::logger().get_filename(str));
}

void annotate_log(std::string str) {
  if (!use_log_) { return; }
  if (log_json) {
    json j;
    j[INSTRUCTION] = "ANNOTATE";
    j[ANNOTATION] = str;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log("# " + str);
  }
}

void toggle_log(bool b) {
  use_log_ = b;
}

// todo: make this a function of Checkpointpool
// should we traverse all externals in chronological order or reverse chronological order?
// my intuition tell me it should be reversed, because the reversed order prioritize the newer external,
// which has tensor more near it unevicted (because of staleness).
// if we go with chronological order, those tensors might be evicted.
void clear_checkpointpool() {
  while (!pool.exts.empty()) {
    if (auto e = pool.exts.back().lock()) {
      e->value->pin();
    }
    pool.exts.pop_back();
  }
}

void unset_memory_budget() {
  pool.has_memory_budget = false;
}

void set_memory_budget(long budget) {
  pool.memory_budget = budget;
  pool.has_memory_budget = true;
}

void toggle_sampling(bool sample) {
  pool.sample_tensors = sample;
}

void toggle_ignore_small_tensors(bool ignore) {
  pool.ignore_small_tensors = ignore;
}

void reset_profile() {
  base_compute_time_ = 0;
  remat_compute_time_ = 0;
  search_time_ = 0;
  cost_time_ = 0;
}

void toggle_profile(bool profile) {
  use_profile_ = profile;
}

long remat_compute_time() {
  return remat_compute_time_;
}

long base_compute_time() {
  return base_compute_time_;
}

long compute_time() {
  return base_compute_time() + remat_compute_time();
}

long cost_time() {
  return cost_time_;
}

long search_time() {
  return search_time_;
}

long loop_time() {
  return search_time() - cost_time();
}
}

// map over the tensor in the ivalue.
template<typename F>
IValue map_ivalue(const F& f, const IValue& iv) {
  if (iv.isTensor()) {
    return f(iv.toTensor());
  } else if (iv.isScalar()) {
    return iv;
  } else {
    TORCH_CHECK(false, "unknown ivalue type: ", *(iv.type()));
    throw;
  }
}

// note: please be deterministic (same input give same output/same mutation on input no matter how many time it is called).
// if it is not deterministic, at least dont change the shape of the output (if input shape not changed).
// otherwise the code will break.
void CheckpointFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "inside fallback" << std::endl;
  std::cout << op.operator_name() << std::endl;
  auto s = op.schema();
  std::cout << s << std::endl;
  size_t num_arg = s.arguments().size(), num_ret = s.returns().size();
  TORCH_CHECK(!s.is_mutable()); // todo: deal with mutability. s.hasAnyAliasInfo() might be useful.
  std::vector<IValue> reversed_in; // popping them from the jit stack and pushing them back will reverse stuff.
  // but should we really reverse stuff? there is a peek() function which doesnt.
  // ezyang seems to want to replace stack impl from std::vector to some sort of array,
  // so slower peek() though.
  for (size_t i = 0; i < num_arg; ++i) {
    reversed_in.push_back(torch::jit::pop(stack));
  }
  // todo: is it safe to save a torch::jit::stack*?
  // todo: modify on heap instead of pushing and popping?
  auto call =
    [=](){
      for (auto it = reversed_in.rbegin(); it != reversed_in.rend(); ++it) {
        torch::jit::push(stack, map_ivalue(native::decheckpoint, *it));
      }
      op.callBoxed(stack);
      std::vector<IValue> reversed_out;
      for (size_t i = 0; i < num_ret; ++i) {
        reversed_out.push_back(torch::jit::pop(stack));
      }
      return reversed_out;
    };
  auto reversed_out = call();
  for (auto it = reversed_out.rbegin(); it != reversed_out.rend(); ++it) {
    torch::jit::push(stack, map_ivalue(native::checkpoint, *it));
  }
  return;

  // rematerializer: grab the reversed_in, uncheckpoint and push all of them onto the stack,
  // grab the output arguments from the stack and push them into reversed_out,
  // then walk through all the tensor to update a list of receivers.

  // as we are initializing, we need to do extra job.
  // in particular, we will save a vector of children in the rematerializer,
  // so when rematerializing, we can plug the value back in.
  // a vector of parents are also saved for fast parent access which is needed for eviction cost evaluation.

  // traverse all the ivalue but map all the tensor inside.
}

// todo: i can also use a torch library impl instead of calling fallback explicitly. should i do that?
struct Register {
  Register() {
    static auto registration = c10::Dispatcher::singleton().registerFallback(DispatchKey::Checkpoint,
                                                                             KernelFunction::makeFromBoxedFunction<&CheckpointFallback>(),
                                                                             "checkpoint");
  }
} register_checkpoint;

}
