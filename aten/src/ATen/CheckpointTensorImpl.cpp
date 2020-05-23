#include <ATen/CheckpointTensorImpl.h>
#include <ATen/Logger.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at {

void AliasPool::evict() {
  TORCH_CHECK(lock_count == 0);
  for (const weak& w : tensors) {
    if (auto cell = w.lock()) {
      cell->evict();
    }
  }
}

void External::release_resources() {
  value->evict();
  value.reset();
}

Tensors stitch(const strongs& input_values,
               const std::vector<std::tuple<Tensor, size_t>>& constants) {
  Tensors input;
  size_t i = 0, j = 0;
  while (i != input_values.size() || j != constants.size()) {
    if (j < constants.size() && std::get<1>(constants[j]) == input.size()) {
      Tensor t = std::get<0>(constants[j]);
      TORCH_CHECK(!t.key_set().has(DispatchKey::CheckpointTensorId));
      input.push_back(t);
      ++j;
    }
    else {
      CHECK(i < input_values.size());
      input.push_back(input_values[i]->get());
      ++i;
    }
  }
  return input;
}

void Rematerializer::remat() {
  // TODO: refactor using RAII for exception safety.
  for (const strong& s : input_values) {
    ++(s->pool->lock_count);
  }
  Tensors ts = stitch(input_values, constants);
  auto ret = func(ts);
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      output_cell->fill(ret[i]);
    }
  }
  for (const strong& s : input_values) {
    --(s->pool->lock_count);
  }
}

namespace native {

Tensor checkpoint(const Tensor& t) {
  auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t.detach());
  DTRLogConstant(cpti->counter_name());
  DTRLogMemory(cpti->counter_name(), cpti->ref->value->value->memory());
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

void new_log(std::string str) {
  DTRLogger::logger().out = std::ofstream(DTRLogger::logger().get_filename(str));
}

void annotate_log(std::string str) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "ANNOTATE";
    j[ANNOTATION] = str;
    DTRLogger::logger().log(j.dump());
  } else {
    DTRLogger::logger().log("# " + str);
  }
}

void clear_checkpointpool() {
  // not implemented yet.
}

}

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  auto ret = intrusive_ptr<CheckpointTensorImpl>::make(ref);
  DTRLogCopy(ret->counter_name(), counter_name());
  return ret;
}

void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(impl->key_set().has(DispatchKey::CheckpointTensorId));
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(impl.get());
  TORCH_CHECK(cpti != nullptr);
  ref->value = cpti->ref->value;
  DTRLogCopyFrom(counter_name(), cpti->counter_name());
}

int CheckpointTensorImpl::counter = 0;

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

// remat take a single vector of tensors,
// while there are two vector, one storing nonconstants and one storing constants.
// the constants are small and they will not be considered for eviction.
// however, we have to stitch the two vectors together to pass it in remat.
// the size_t in constants decide the location to stitch them in, while input_values fill in the rest.
MakeRawResult make_raw(const rematerialize_function_t& remat_f,
                       // We need this to assign alias pool.
                       // This is ugly as fuck but after refactoring we dont even need stitching anymore.
                       const Tensors& raw_input,
                       const strongs& input_values,
                       const std::vector<std::tuple<Tensor, size_t>>& constants) {
  Tensors inputs = stitch(input_values, constants);
  time_t pre = std::chrono::system_clock::now();
  auto outputs_raw = remat_f(inputs);
  time_t post = std::chrono::system_clock::now();
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), input_values, constants, remat_f);
  for (const Tensor& t : outputs_raw) {
    int alias = get_alias(inputs, t);
    intrusive_ptr<AliasPool> pool;
    if (alias == -1) {
      pool = intrusive_ptr<AliasPool>::make(Unsafe(), true, memory(t));
    }
    else if (auto* cpti = dynamic_cast<CheckpointTensorImpl*>(raw_input[alias].unsafeGetTensorImpl())) {
      pool = cpti->ref->value->value->pool;
    } else { // alias to an constant. unevictable.
      pool = intrusive_ptr<AliasPool>::make(Unsafe(), false, memory(t));
    }
    auto e = intrusive_ptr<External>::make(t, pool, remat);
    pool->tensors.push_back(weak(e->value));
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }
  remat->outputs = weak_outputs;
  return {outputs, aliases, post - pre, remat};
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const Tensors& inputs) {
  strongs input_values;
  std::vector<std::tuple<Tensor, size_t>> constants;
  std::vector<size_t> constant_idx;
  std::vector<std::string> args;
  for (const Tensor& t: inputs) {
    if (auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl())) {
      input_values.push_back(cpti->ref->value->value);
      args.push_back(cpti->counter_name());
    }
    else {
      size_t idx = input_values.size() + constants.size();
      constants.push_back({t, idx});
      constant_idx.push_back(idx);
    }
  }
  std::vector<std::string> res;
  auto ret = make_raw(remat, inputs, input_values, constants);
  Tensors tensors;
  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
    res.push_back(get_cpti(cp)->counter_name());
  }
  DTRLogCall(res, name, args, constant_idx, from_time(ret.time));
  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor t = tensors[i];
    auto cpti = get_cpti(t);
    DTRLogMemory(cpti->counter_name(), cpti->ref->value->value->memory());
    DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
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
  strongs input_values;
  std::vector<std::tuple<Tensor, size_t>> constants;
  std::vector<size_t> constant_idx;
  std::vector<std::string> args;
  for (const Tensor& t: inputs) {
    if (auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl())) {
      input_values.push_back(cpti->ref->value->value);
      args.push_back(cpti->counter_name());
    }
    else {
      size_t idx = input_values.size() + constants.size();
      constants.push_back({t, idx});
      constant_idx.push_back(idx);
    }
  }
  auto ret = make_raw(remat, inputs, input_values, constants);
  const auto& modified = ret.outputs;
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
  DTRLogMutate(name, args, constant_idx, mutate_idx, from_time(ret.time));
}

void CheckpointTensorImpl::release_resources() {
  DTRLogRelease(counter_name());
  ref.reset();
}

}
