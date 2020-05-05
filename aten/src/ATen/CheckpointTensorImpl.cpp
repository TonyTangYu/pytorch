#include <ATen/CheckpointTensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <../../../third_party/json/single_include/nlohmann/json.hpp>

namespace at {

struct DTRLogger {
  std::string time_prefix;
  std::ofstream out;
  static std::string get_time_prefix() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    return
      std::to_string(1900+tm->tm_year) + "-" +
      std::to_string(1+tm->tm_mon) + "-" +
      std::to_string(tm->tm_mday) + "-" +
      std::to_string(tm->tm_hour) + "-" +
      std::to_string(tm->tm_min) + "-" +
      std::to_string(tm->tm_sec);
  }
  std::string get_filename(const std::string& name) {
    return time_prefix + "-" + name + ".log";
  }
  DTRLogger() : time_prefix(get_time_prefix()), out(get_filename("default")) { }
  void log(const std::string& str) {
    out << str << std::endl;
  }
};

static DTRLogger logger;

using json = nlohmann::json;
bool log_json = true;
const std::string INSTRUCTION = "INSTRUCTION";
const std::string ANNOTATION = "ANNOTATION";
const std::string RELEASE = "RELEASE";
const std::string TIME = "TIME";
const std::string ARGS = "ARGS";
const std::string MEMORY = "MEMORY";
const std::string NAME = "NAME";
const std::string CONSTANT = "CONSTANT";
const std::string CONSTANTS = "CONSTANTS";

void DTRLogConstant(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    logger.log(j.dump());
  } else {
    logger.log(CONSTANT + " " + name);
  }
}

void DTRLogMemory(const std::string& name, size_t memory) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = MEMORY;
    j[NAME] = name;
    j[MEMORY] = std::to_string(memory);
    logger.log(j.dump());
  } else {
    logger.log(name + " " + MEMORY + ": " + std::to_string(memory));
  }
}

namespace native {

Tensor checkpoint(const Tensor& t) {
  auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t.detach());
  DTRLogConstant(cpti->counter_name());
  DTRLogMemory(cpti->counter_name(), cpti->ref->value->memory());
  return Tensor(cpti);
}

Tensor decheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  return cpti->ref->value->t;
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

void new_log(std::string str) {
  logger.out = std::ofstream(logger.get_filename(str));
}

void annotate_log(std::string str) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "ANNOTATE";
    j[ANNOTATION] = str;
    logger.log(j.dump());
  } else {
    logger.log(str);
  }
}

}

void DTRLogCopy(const std::string& new_name, const std::string& old_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY";
    j["DST"] = new_name;
    j["SRC"] = old_name;
    logger.log(j.dump());
  } else {
    logger.log(new_name + " = " + old_name);
  }
}

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  auto ret = intrusive_ptr<CheckpointTensorImpl>::make(ref);
  DTRLogCopy(ret->counter_name(), counter_name());
  return ret;
}

int CheckpointTensorImpl::counter = 0;

Tensor checkpoint_raw(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t.detach()));
}

// remat take a single vector of tensors,
// while there are two vector, one storing nonconstants and one storing constants.
// the constants are small and they will not be considered for eviction.
// however, we have to stitch the two vectors together to pass it in remat.
// the size_t in constants decide the location to stitch them in, while input_values fill in the rest.
std::tuple<Tensors, duration_t> make_raw(const rematerialize_function_t& remat,
                                         const strongs& input_values,
                                         const std::vector<std::tuple<Tensor, size_t>>& constants) {
  std::vector<Tensor> input;
  size_t i = 0, j = 0;
  while (i != input_values.size() || j != constants.size()) {
    if (j < constants.size() && std::get<1>(constants[j]) == input.size()) {
      input.push_back(std::get<0>(constants[j]));
      ++j;
    }
    else {
      CHECK(i < input_values.size());
      CHECK(!input_values[i]->t.key_set().has(DispatchKey::CheckpointTensorId));
      input.push_back(input_values[i]->t);
      ++i;
    }
  }
  time_t pre = std::chrono::system_clock::now();
  auto output = remat(input);
  time_t post = std::chrono::system_clock::now();
  return {output, post - pre};
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

void DTRLogCall(const std::vector<std::string>& res,
                const std::string& name,
                const std::vector<std::string>& args,
                const std::vector<size_t>& constants,
                const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "CALL";
    j[NAME] = name;
    j["RESULT"] = res;
    j[ARGS] = args;
    j[CONSTANTS] = constants;
    j[TIME] = time;
    logger.log(j.dump());
  } else {
    CHECK(constants.size() == 0); //TODO: implement.
    std::string arg = name + "(";
    for (const auto& s : arg) {
      arg += s;
      arg += ", ";
    }
    arg += ")";
    std::string log = "(";
    for (const auto& s: res) {
      log += s;
      log += ", ";
    }
    log += ") = ";
    log += arg;
    log += " TIME: ";
    log += time;
    logger.log(log);
  }
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const Tensors& input) {
  strongs input_values;
  std::vector<std::tuple<Tensor, size_t>> constants;
  std::vector<size_t> constant_idx;
  std::vector<std::string> args;
  for (const Tensor& t: input) {
    if (auto* cpt = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl())) {
      input_values.push_back(cpt->ref->value);
      args.push_back(cpt->counter_name());
    }
    else {
      size_t idx = input_values.size() + constants.size();
      constants.push_back({t, idx});
      constant_idx.push_back(idx);
    }
  }
  std::vector<std::string> res;
  auto ret = make_raw(remat, input_values, constants);
  Tensors tensors;
  for (const Tensor& t: std::get<0>(ret)) {
    auto cp = checkpoint_raw(t);
    tensors.push_back(cp);
    res.push_back(get_cpti(cp)->counter_name());
  }
  DTRLogCall(res, name, args, constant_idx, from_time(std::get<1>(ret)));
  for (const Tensor& t: tensors) {
    auto cpti = get_cpti(t);
    DTRLogMemory(cpti->counter_name(), cpti->ref->value->memory());
  }
  return tensors;
}

void DTRLogMutate(const std::string& name,
                  const std::vector<std::string>& args,
                  const std::vector<size_t>& constants,
                  const std::vector<size_t>& mutate,
                  const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "MUTATE";
    j[NAME] = name;
    j[ARGS] = args;
    j[CONSTANTS] = constants;
    j["MUTATE"] = mutate;
    j[TIME] = time;
    logger.log(j.dump());
  } else {
    CHECK(constants.size() == 0); //TODO: implement.
    std::string log = name;
    log += "(";
    for (const auto& s : args) {
      log += s;
      log += ", ";
    }
    log += ") ";
    log += " MUTATING: ";
    log += "(";
    for (const size_t i : mutate) {
      log += std::to_string(i);
      log += ", ";
    }
    log += ") ";
    log += TIME;
    log += ": ";
    log += time;
    logger.log(log);
  }
}

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
    if (auto* cpt = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl())) {
      input_values.push_back(cpt->ref->value);
      args.push_back(cpt->counter_name());
    }
    else {
      size_t idx = input_values.size() + constants.size();
      constants.push_back({t, idx});
      constant_idx.push_back(idx);
    }
  }
  auto ret = make_raw(remat, input_values, constants);
  const auto& modified = std::get<0>(ret);
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = intrusive_ptr<CheckpointTensorCell>::make(modified[idx]);
  }
  DTRLogMutate(name, args, constant_idx, mutate_idx, from_time(std::get<1>(ret)));
}

void DTRLogRelease(const std::string& counter_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = RELEASE;
    j[NAME] = counter_name;
    logger.log(j.dump());
  } else {
    logger.log(RELEASE + ": " + counter_name);
  }
}

void CheckpointTensorImpl::release_resources() {
  DTRLogRelease(counter_name());
    ref.reset();
}

}
