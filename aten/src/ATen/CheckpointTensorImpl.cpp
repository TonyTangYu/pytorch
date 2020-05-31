#include <ATen/CheckpointTensorImpl.h>
#include <ATen/Logger.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at {

CheckpointPool pool;

long current_memory() {
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  return device_stat.allocated_bytes[0].current;
}

void checkpoint_auto_evict() {
  pool.auto_evict();
}

void CheckpointPool::auto_evict() {
  if (has_memory_budget) {
    while (current_memory() > memory_budget) {
      evict();
    }
  }
}

void CheckpointPool::evict() {
  TORCH_CHECK(aps.size() > 0);
  bool shrinked = false;
  int evict_idx = -1;
  double evict_score = INFINITY;
  time_t current_time = std::chrono::system_clock::now();
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  for (size_t i = 0; i < aps.size();) {
    auto cannot_evict = [&]() {
                          shrinked = true;
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();
    if (!ap_strong.defined()) {
      cannot_evict();
    } else {
      if (ap_strong->evictable()) {
        double score = ap_strong->score(current_time);
        if (score < evict_score) {
          evict_score = score;
          evict_idx = i;
        }
      }
      ++i;
    }
  }
  if (evict_idx == -1) {
    TORCH_CHECK(shrinked);
  } else {
    auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                            ap_strong->evict();
                            remove_from_aps(evict_idx);
                          };
    evict_from_idx(evict_idx);
  }
}

CheckpointPool::CheckpointPool() {
  c10::set_evict_func(checkpoint_auto_evict);
}

bool use_log = true;

namespace native {

Tensor checkpoint(const Tensor& t) {
  auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t.detach());
  if (use_log) {
    DTRLogConstant(cpti->counter_name());
    DTRLogMemory(cpti->counter_name(), cpti->ref->value->value->memory());
  }
  return Tensor(cpti);
}

Tensor uncheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  return cpti->ref->value->value->get();
}

void pin(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  cpti->ref->value->value->pin();
}

Tensor decheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti ? cpti->ref->value->value->get() : t;
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor try_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? t : checkpoint(t);
}

void new_log(std::string str) {
  DTRLogger::logger().out = std::ofstream(DTRLogger::logger().get_filename(str));
}

void annotate_log(std::string str) {
  if (!use_log) { return; }
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
  use_log = b;
}

void clear_checkpointpool() {
  // not implemented yet.
}

void unset_memory_budget() {
  pool.has_memory_budget = false;
}

void set_memory_budget(long budget) {
  pool.memory_budget = budget;
  pool.has_memory_budget = true;
}

}

Tensor uncheckpoint(const strong& input) {
  return input->get();
}

Tensors uncheckpoint(const strongs& inputs) {
  Tensors ret;
  for (const strong& input : inputs) {
    ret.push_back(uncheckpoint(input));
  }
  return ret;
};

Tensors try_checkpoint(const Tensors& inputs) {
  Tensors ret;
  for (const Tensor& input : inputs) {
    ret.push_back(at::native::try_checkpoint(input));
  }
  return ret;
}

CheckpointInfo merge_cpi(CheckpointInfo l, CheckpointInfo r) {
  return CheckpointInfo(l.compute_cost + r.compute_cost,
                        std::max(l.last_used_time, r.last_used_time));
}

void AliasPool::evict() {
  TORCH_CHECK(!ecn);
  ecn = head_remat->get_ecn(last_used_time);
  auto ecns = neighbor_ecn();
  for (const auto& necn : ecns) {
    merge<CheckpointInfo>(merge_cpi, ecn, necn);
  }
  auto b4 = current_memory();
  TORCH_CHECK(memory > 0);
  TORCH_CHECK(lock_count == 0);
  TORCH_CHECK(!is_evicted);
  is_evicted = true;
  for (const weak& w : tensors) {
    if (auto cell = w.lock()) {
      cell->evict();
    }
  }
  // TORCH_CHECK(current_memory() < b4);
  // somehow it is still evicting unevictable stuff.
}

double AliasPool::score(time_t current_time) {
  auto cpi = head_remat->get_cpi(last_used_time);
  auto ecns = neighbor_ecn();
  for (const auto& necn : ecns) {
    cpi = merge_cpi(cpi, get_t(necn));
  }
  return cpi.score(memory, current_time);
}

void External::release_resources() {
  value->evict();
  value.reset();
}

void Rematerializer::remat() {
  // TODO: refactor using RAII for exception safety.
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors ts = uncheckpoint(inputs);
  auto ret = func(ts);
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

ecn_ptr Rematerializer::get_ecn(time_t last_used_time) {
  if (ecn) {
    auto cpi = get_t(ecn);
    update_t(ecn, CheckpointInfo(cpi.compute_cost, std::max(last_used_time, cpi.last_used_time)));
  } else {
    ecn = ecn_ptr::make(CheckpointInfo(compute_cost, last_used_time));
  }
  return ecn;
}

CheckpointInfo Rematerializer::get_cpi(time_t last_used_time) {
  return CheckpointInfo(ecn ? duration_t(0) : compute_cost, last_used_time);
}

std::vector<ecn_ptr> AliasPool::neighbor_ecn() {
  std::vector<ecn_ptr> ret;
  for (size_t i = 0; i < neighbors.size();) {
    if (auto cptc = neighbors[i].lock()) {
      if (cptc->pool->ecn) {
        ret.push_back(cptc->pool->ecn);
      }
      ++i;
    } else {
      neighbors[i] = neighbors[neighbors.size() - 1];
      neighbors.pop_back();
    }
  }
  std::sort(ret.begin(), ret.end());
  ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
  return ret;
}

void AliasPool::set_not_evicted(const intrusive_ptr<AliasPool>& self) {
  if (is_evicted) {
    is_evicted = false;
    if (ecn) {
      TORCH_CHECK(head_remat);
      auto cpi = get_t(ecn);
      update_t(ecn, CheckpointInfo(cpi.compute_cost - head_remat->compute_cost, cpi.last_used_time));
      ecn.reset();
    }
    pool.aps.push_back(weak_intrusive_ptr<AliasPool>(self));
  }
}

void CheckpointTensorCell::fill(const Tensor& t) {
  if (!(this->t)) {
    this->t = std::make_unique<Tensor>(t.detach());
    pool->set_not_evicted(pool);
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

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  auto ret = intrusive_ptr<CheckpointTensorImpl>::make(ref);
  if (use_log) {
    DTRLogCopy(ret->counter_name(), counter_name());
  }
  return ret;
}

void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(impl->key_set().has(DispatchKey::CheckpointTensorId));
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(impl.get());
  TORCH_CHECK(cpti != nullptr);
  ref->value = cpti->ref->value;
  if (use_log) {
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

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
};

void add_neighbor(const strong& l, const strong& r) {
  l->pool->neighbors.push_back(weak(r));
}

// remat take a single vector of tensors,
// while there are two vector, one storing nonconstants and one storing constants.
// the constants are small and they will not be considered for eviction.
// however, we have to stitch the two vectors together to pass it in remat.
// the size_t in constants decide the location to stitch them in, while input_values fill in the rest.
MakeRawResult make_raw(const rematerialize_function_t& remat_f,
                       const strongs& inputs) {
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors raw_inputs = uncheckpoint(inputs);
  time_t pre = std::chrono::system_clock::now();
  auto raw_outputs = remat_f(raw_inputs);
  time_t post = std::chrono::system_clock::now();
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
      if (m > 0) {
        pool.aps.push_back(weak_intrusive_ptr<AliasPool>(alias_pool));
      }
    }
    else {
      alias_pool = inputs[alias]->pool;
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += (post - pre);
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat);
    alias_pool->tensors.push_back(weak(e->value));
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }
  remat->outputs = weak_outputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (is_alias(raw_inputs[i], raw_outputs[j])) {
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
  strongs input_values;
  std::vector<std::string> args;
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->ref->value->value);
    if (use_log) {
      args.push_back(cpti->counter_name());
    }
  }
  std::vector<std::string> res;
  auto ret = make_raw(remat, input_values);
  Tensors tensors;
  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
    res.push_back(get_cpti(cp)->counter_name());
  }
  if (use_log) {
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

// TODO: check that mutated value does not have alias.
void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  const Tensors& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                   new_input_values[idx] = t[idx].clone();
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  strongs input_values;
  std::vector<std::string> args;
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->ref->value->value);
    if (use_log) {
      args.push_back(cpti->counter_name());
    }
  }
  auto ret = make_raw(remat, input_values);
  const auto& modified = ret.outputs;
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
  if (use_log) {
    DTRLogMutate(name, args, mutate_idx, from_time(ret.time));
  }
}

void CheckpointTensorImpl::release_resources() {
  if (use_log) {
    DTRLogRelease(counter_name());
  }
  ref.reset();
}

}
