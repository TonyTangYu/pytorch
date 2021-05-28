#include <ATen/CheckpointTensorImpl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <signal.h>

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

void CheckpointPool::auto_evict() {
  if (has_memory_budget) {
    while (current_memory() > memory_budget) {
      evict();
    }
  }
}

void CheckpointPool::evict() {
  time_t pre = std::chrono::system_clock::now();
  TORCH_CHECK(aps.size() > 0);
  // shrunk: either something has been evicted or the pools have gotten smaller
  bool shrunk = false;
  int evict_idx = -1;
  double evict_cost = INFINITY;
  time_t current_time = std::chrono::system_clock::now();
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  std::uniform_int_distribution<> distrib(1, 1 * std::max(1, static_cast<int>(std::sqrt(aps.size()))));
  // sampling a random independent subset of all evictable tensors to find the cheapest tensor to evict.
  for (size_t i = 0; i < aps.size();) {
    auto cannot_evict = [&]() {
                          shrunk = true;
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();
    if (!ap_strong.defined()) {
      cannot_evict();
    }
    else if (ap_strong->ecn) {
      cannot_evict();
    }
    else {
      if (ap_strong->evictable()) {
        double cost = ap_strong->cost(current_time);
        if (cost < evict_cost) {
          evict_cost = cost;
          evict_idx = i;
        }
      }

      if (sample_tensors) {
        i += distrib(gen);
      } else {
        i += 1;
      }
    }
  }
  if (evict_idx == -1) {
    TORCH_CHECK(shrunk);
  } else {
    auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                            ap_strong->evict();
                            remove_from_aps(evict_idx);
                          };
    evict_from_idx(evict_idx);
  }
  time_t post = std::chrono::system_clock::now();
  search_time_ += (post - pre).count();
}

// todo: make this a function of Checkpointpool
// should we traverse all externals in chronological order or reverse chronological order?
// my intuition tell me it should be reversed, because the reversed order prioritize the newer external,
// which has tensor more near it unevicted (because of staleness).
// if we go with chronological order, those tensors might be evicted.
void CheckpointPool::clear_checkpointpool() {
  while (!exts.empty()) {
    if (auto e = exts.back().lock()) {
      e->value->pin();
    }
    exts.pop_back();
  }
  aps.clear();
}

Tensor uncheckpoint(const strong& input) {
  return input->get();
}

Tensors uncheckpoint(const strongs& inputs) {
  Tensors ret;
  ret.reserve(inputs.size());
  for (const strong& input : inputs) {
    ret.push_back(uncheckpoint(input));
  }
  return ret;
};

Tensors try_checkpoint(const Tensors& inputs) {
  Tensors ret;
  ret.reserve(inputs.size());
  for (const Tensor& input : inputs) {
    ret.push_back(at::native::try_checkpoint(input));
  }
  return ret;
}

void Rematerializer::remat() {
  // TODO: refactor using RAII for exception safety.
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors ts = uncheckpoint(inputs);
  time_t pre = std::chrono::system_clock::now();
  auto ret = func(ts);
  time_t post = std::chrono::system_clock::now();
  pool.auto_evict();
  remat_compute_time_ += (post - pre).count();
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      output_cell->fill(ret[i]);
    }
  }
  ecn.reset();
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
}

ecn_ptr Rematerializer::get_ecn() {
  if (!ecn) {
    ecn = ecn_ptr::make(CheckpointInfo(compute_cost));
  }
  return ecn;
}

CheckpointInfo merge_cpi(CheckpointInfo l, CheckpointInfo r) {
  return CheckpointInfo(l.compute_cost + r.compute_cost);
}

std::set<ecn_ptr> AliasPool::neighbor_ecn() {
  std::set<ecn_ptr> ptr_set;
  int size = neighbors.size();
  for (size_t i = 0; i < size;) {
    if (auto cptc = neighbors[i].lock()) {
      if (cptc->pool->ecn) {
        ptr_set.insert(cptc->pool->ecn);
      }
      ++i;
    } else {
      neighbors[i] = neighbors[size - 1];
      --size;
    }
  }
  if (size < neighbors.size()) {
    neighbors.erase(neighbors.begin() + size);
  }
  return ptr_set;
}


double AliasPool::cost(time_t current_time) {
  TORCH_CHECK(evictable());
  time_t pre = std::chrono::system_clock::now();
  auto cpi = CheckpointInfo(head_remat->compute_cost);
  auto ecns = neighbor_ecn();
  for (const auto& necn : ecns) {
    cpi = merge_cpi(cpi, get_t(necn));
  }
  auto ret = cpi.cost(memory, (current_time - last_used_time).count());
  time_t post = std::chrono::system_clock::now();
  cost_time_ += (post - pre).count();
  return ret;
}

void AliasPool::evict() {
  TORCH_CHECK(!ecn);
  ecn = head_remat->get_ecn();
  auto ecns = neighbor_ecn();
  for (const auto& necn : ecns) {
    merge<CheckpointInfo>(merge_cpi, ecn, necn);
  }
  TORCH_CHECK(lock_count == 0);
  for (const weak& w : tensors) {
    if (auto cell = w.lock()) {
      cell->evict();
    }
  }
}

void AliasPool::set_not_evicted(const intrusive_ptr<AliasPool>& self) {
  if (ecn) {
    TORCH_CHECK(head_remat);
    auto cpi = get_t(ecn);
    update_t(ecn, CheckpointInfo(cpi.compute_cost - head_remat->compute_cost));
    ecn.reset();
    pool.add(self);
  }
}

void CheckpointTensorCell::fill(const Tensor& t) {
  if (!(this->t)) {
    TORCH_CHECK(!at::native::is_checkpoint(t));
    TORCH_CHECK(!t.key_set().has(DispatchKey::Checkpoint))
    this->t = std::make_unique<Tensor>(t.detach());
    pool->set_not_evicted(pool);
    if (!defined) {
      defined = true;
      is_undefined_tensor = !t.defined();
      key_set_ = t.key_set();
      if (t.requires_grad()) {
        key_set_ = key_set_.add(DispatchKey::Autograd);
      }
      dtype_ = t.dtype();
      optional_device_ = t.optional_device();
    }
  }
}

Tensor CheckpointTensorImpl::get() const {
  return ref->value->value->get();
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
  // I was once a smartasss and thought I didnt need to copy,
  // for the value is immutable.
  // Turnout I am a dumbass:
  // the autogradmeta is mutable.
  auto ret = intrusive_ptr<CheckpointTensorImpl>::make(ref);
  if (use_log_) {
    DTRLogCopy(ret->counter_name(), counter_name());
  }
  return ret;
}

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(VariableVersion&& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
}

void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(key_set() == impl->key_set());
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

void add_neighbor(const strong& l, const strong& r) {
  l->pool->neighbors.push_back(weak(r));
  r->pool->neighbors.push_back(weak(l));
}

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
};

MakeRawResult make_raw(const rematerialize_function_t& remat_f,
                       const strongs& inputs) {
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors raw_inputs = uncheckpoint(inputs);
  time_t pre = std::chrono::system_clock::now();
  auto raw_outputs = remat_f(raw_inputs);
  time_t post = std::chrono::system_clock::now();
  pool.auto_evict();
  base_compute_time_ += (post - pre).count();
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, post - pre);

  for (const Tensor& t : raw_outputs) {
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);
    if (alias == -1) {
      auto m = memory(t);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m);
      pool.add(alias_pool);
    }
    else {
      alias_pool = inputs[alias]->pool;
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += (post - pre);
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat);
    pool.exts.push_back(weak_intrusive_ptr<External>(e));
    alias_pool->tensors.push_back(weak(e->value));
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }
  remat->outputs = weak_outputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        add_neighbor(inputs[i], outputs[j]->value);
      }
    }
  }
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
  return {outputs, aliases, post - pre, remat};
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const Tensors& inputs) {
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  auto input_size = checkpointed_inputs.size();

  strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = must_get_cpti(t);
    input_values.push_back(cpti->ref->value->value);
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
  }

  auto ret = make_raw(remat, input_values);

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
  }

  if (use_log_) {
    std::vector<std::string> res;
    res.reserve(ret.outputs.size());

    for (const auto& tensor : tensors) {
      res.push_back(get_cpti(tensor)->counter_name());
    }

    DTRLogCall(res, name, args, from_time(ret.time));
    for (size_t i = 0; i < tensors.size(); ++i) {
      Tensor t = tensors[i];
      auto cpti = get_cpti(t);
      DTRLogMemory(cpti->counter_name(), cpti->ref->value->value->memory());
      DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
    }
  }

  return tensors;
}

void CheckpointTensorImpl::release_resources() {
  if (use_log_) {
    DTRLogRelease(counter_name());
  }
  ref.reset();
}

struct CheckpointFunctionsImpl: CheckpointFunctions {
  void new_log(std::string str) override {
    DTRLogger::logger().out = std::ofstream(DTRLogger::logger().get_filename(str));
  }
  void annotate_log(std::string str) override {
    if (use_log_) {
      json j;
      j[INSTRUCTION] = "ANNOTATE";
      j[ANNOTATION] = str;
      DTRLogger::logger().log(j.dump());
    }
  }
  void toggle_log(bool b) override {
    use_log_ = b;
  }
  void clear_checkpointpool() override {
    pool.clear_checkpointpool();
  }
  void unset_memory_budget() override {
    pool.has_memory_budget = false;
  }
  void set_memory_budget(long budget) override {
    pool.memory_budget = budget;
    pool.has_memory_budget = true;
  }
  void toggle_sampling(bool sample) override {
    pool.sample_tensors = sample;
  }
  void toggle_ignore_small_tensors(bool ignore) override {
    pool.ignore_small_tensors = ignore;
  }
  void toggle_profile(bool profile) override {
    use_profile_ = profile;
  }
  void reset_profile() override {
    base_compute_time_ = 0;
    remat_compute_time_ = 0;
    search_time_ = 0;
    cost_time_ = 0;
  }
  long base_compute_time() override {
    return base_compute_time_;
  }
  long remat_compute_time() override {
    return remat_compute_time_;
  }
  long compute_time() override {
    return base_compute_time() + remat_compute_time();
  }
  long cost_time() override {
    return cost_time_;
  }
  long search_time() override {
    return search_time_;
  }
  long loop_time() override {
    return search_time() - cost_time();
  }
};

CheckpointFunctions* GetCheckpointFunctions() {
  static CheckpointFunctionsImpl cpfi;
  return &cpfi;
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
  must_get_cpti(t)->ref->value->value->pin();
}

bool is_checkpoint(const Tensor& t) {
  return get_cpti(t) != nullptr;
}

Tensor try_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? t : checkpoint(t);
}

}

// map over the tensor in the ivalue.
// weird stuff. seems like i cant write a generic function over all list :(
template<typename F>
IValue map_ivalue(const F& f, const IValue& iv) {
  if (iv.isTensor()) {
    return f(iv.toTensor());
  } else if (iv.isScalar() || iv.isBool() || iv.isDevice() || iv.isNone() || iv.isIntList() || iv.isBoolList() || iv.isDoubleList()) {
    return iv;
  } else if (iv.isTensorList()) {
    std::vector<Tensor> ts;
    for (const auto& t: iv.toTensorList()) {
      ts.push_back(f(t));
    }
    return ts;
  }
  else {
    TORCH_CHECK(false, "unknown ivalue type: ", *(iv.type()));
    throw;
  }
}

Ref<intrusive_ptr<External>> cell_from_tensor(const Tensor& t) {
  return must_get_cpti(t)->ref;
}

// note: please be deterministic (same input give same output/same mutation on input no matter how many time it is called).
// if it is not deterministic, at least dont change the shape of the output (if input shape not changed).
// otherwise the code will break.
// Right now uncheckedpointed tensor is converted into checkpoint tensor before going into CheckpointTensorImpl::make.
// It seems like you can not convert to save time, but it break our logging code.
// the code is a bit cleaner this way, and this extra information maybe helpful.
// So there is two interface: CheckpointTensor's pure Tensors -> Tensors interface, and a stack mutating interface.
// We have to convert twice.
// In particular, we implement a stack mutating interface for checkpointedtensor,
// by implementing a Tensors -> Tensors interface for ordinary tensor (the conversion is handled by tensors::make).
// we implement that by converting it back to stack mutation.
// Additionally, since the stack contain IValue instead of Tensors,
// we have to inject/extracted the Tensors to/from the saved IValue
// everytime we convert Tensors to/from stack.
// Reminder: you can convert IValue to/from Tensor, but you should not do that in here,
// as IValue may hold zero or more Tensor.
// the only way to construct/destruct an IValue should be map_ivalue.
void CheckpointFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  size_t before_size = stack->size();
  auto s = op.schema();
  // std::cout << s << std::endl;
  size_t num_arg = s.arguments().size();
  // todo: use s.hasAnyAliasInfo() to figure out alias info instead of doing a runtime loop.
  std::vector<IValue> checkpoint_reversed_ivalue_in; // popping them from the jit stack and pushing them back will reverse stuff.
  std::vector<bool> checkpoint_reversed_ivalue_in_mutable;
  // but should we really reverse stuff? there is a peek() function which doesnt.
  // ezyang seems to want to replace stack impl from std::vector to some sort of list,
  // so slower peek() though.
  for (size_t i = 0; i < num_arg; ++i) {
    checkpoint_reversed_ivalue_in.push_back(torch::jit::pop(stack));
    const auto& aliasInfo = s.arguments()[s.arguments().size() - 1 - i].alias_info();
    checkpoint_reversed_ivalue_in_mutable.push_back(aliasInfo && aliasInfo.value().isWrite());
  }
  Tensors original_tensors_in;
  strongs checkpoint_tensors_in;
  std::vector<bool> checkpoint_tensors_in_mutable;
  auto it = checkpoint_reversed_ivalue_in.rbegin();
  auto mit = checkpoint_reversed_ivalue_in_mutable.rbegin();
  while (it != checkpoint_reversed_ivalue_in.rend()) {
    TORCH_CHECK(mit != checkpoint_reversed_ivalue_in_mutable.rend());
    map_ivalue([&](const Tensor& t) {
                 original_tensors_in.push_back(t);
                 auto tcp = native::try_checkpoint(t);
                 auto* cpti = must_get_cpti(tcp);
                 checkpoint_tensors_in.push_back(cpti->ref->value->value);
                 checkpoint_tensors_in_mutable.push_back(*mit);
                 return t; // dont care value
               }, *it);
    ++it;
    ++mit;
  }
  // todo: modify on heap instead of pushing and popping?
  struct JitRemat {
    struct Boxed {
      c10::OperatorHandle op;
      std::vector<IValue> checkpoint_reversed_ivalue_in;
      std::vector<IValue> checkpoint_reversed_ivalue_out;
      std::vector<bool> checkpoint_tensors_in_mutable;
      bool initial_call = true;
      Boxed(const c10::OperatorHandle& op,
            const std::vector<IValue>& checkpoint_reversed_ivalue_in,
            const std::vector<bool>& checkpoint_tensors_in_mutable) :
        op(op),
        checkpoint_reversed_ivalue_in(checkpoint_reversed_ivalue_in),
        checkpoint_tensors_in_mutable(checkpoint_tensors_in_mutable) { }
      Tensors operator()(const Tensors& remat_in) {
        torch::jit::Stack stack;
        size_t count = 0;
        Tensors copied_values;
        for (auto it = checkpoint_reversed_ivalue_in.rbegin(); it != checkpoint_reversed_ivalue_in.rend(); ++it) {
          torch::jit::push(&stack,
                           map_ivalue([&](const Tensor&) {
                                        auto rem_at = remat_in.at(count);
                                        auto ret = [&]() {
                                                     if (checkpoint_tensors_in_mutable.at(count)) {
                                                       auto cloned = rem_at.clone();
                                                       copied_values.push_back(cloned);
                                                       return cloned;
                                                     } else {
                                                       return rem_at;
                                                     }
                                                   }();
                                        ++count;
                                        return ret;
                                      }, *it));
        }
        TORCH_CHECK(count == remat_in.size());
        op.callBoxed(&stack);
        Tensors remat_out;
        auto s = op.schema();
        size_t num_ret = s.returns().size();
        for (size_t i = 0; i < num_ret; ++i) {
          checkpoint_reversed_ivalue_out.push_back(torch::jit::pop(&stack));
        }
        for (auto it = checkpoint_reversed_ivalue_out.rbegin(); it != checkpoint_reversed_ivalue_out.rend(); ++it) {
          map_ivalue([&](const Tensor& t) {
                       remat_out.push_back(t);
                       return t; // dont care value
                     }, *it);
        }
        for (const Tensor& t: copied_values) {
          remat_out.push_back(t);
        }
        if (initial_call) {
          initial_call = false;
        } else {
          checkpoint_reversed_ivalue_out.clear();
        }
        initial_call = false;
        TORCH_CHECK(stack.empty());
        return remat_out;
      }
    };
    std::shared_ptr<Boxed> boxed;
    JitRemat(const c10::OperatorHandle& op,
             const std::vector<IValue>& checkpoint_reversed_ivalue_in,
             const std::vector<bool>& checkpoint_tensors_in_mutable) :
      boxed(std::make_shared<Boxed>(op, checkpoint_reversed_ivalue_in, checkpoint_tensors_in_mutable)) { }
    Tensors operator()(const Tensors& remat_in) { return (*boxed)(remat_in); }
  } remat(op, checkpoint_reversed_ivalue_in, checkpoint_tensors_in_mutable);
  auto make_raw_result = make_raw(remat, checkpoint_tensors_in);
  size_t count = 0;
  for (auto it = remat.boxed->checkpoint_reversed_ivalue_out.rbegin(); it != remat.boxed->checkpoint_reversed_ivalue_out.rend(); ++it) {
    torch::jit::push(stack,
                     map_ivalue([&](const Tensor&) {
                                  auto out = make_raw_result.outputs.at(count);
                                  ++count;
                                  return Tensor(intrusive_ptr<CheckpointTensorImpl>::make(out));
                                }, *it));
  }
  for (size_t i = 0; i < checkpoint_tensors_in.size(); ++i) {
    if (checkpoint_tensors_in_mutable.at(i)) {
      cell_from_tensor(original_tensors_in.at(i))->value = make_raw_result.outputs.at(count);
      ++count;
    }
  }
  // clear the stored ivalue output, so the tensor returned can actually be freed from memory (if evicted).
  remat.boxed->checkpoint_reversed_ivalue_out.clear();
  TORCH_CHECK(before_size - s.arguments().size() + s.returns().size() == stack->size());
  TORCH_CHECK(count == make_raw_result.outputs.size());
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
