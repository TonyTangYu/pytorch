#include <ATen/CheckpointTensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <../../../third_party/json/single_include/nlohmann/json.hpp>

namespace at {

struct DTRLogger {
  std::ofstream out;
  static std::string get_filename() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    std::string str =
      std::to_string(1900+tm->tm_year) + "-" +
      std::to_string(1+tm->tm_mon) + "-" +
      std::to_string(tm->tm_mday) + "-" +
      std::to_string(tm->tm_hour) + "-" +
      std::to_string(tm->tm_min) + "-" +
      std::to_string(tm->tm_sec) + ".log";
    return str;
  }
  DTRLogger() : out(get_filename()) { }
};

void DTRLog(const std::string& str) {
  static DTRLogger logger;
  logger.out << str << std::endl;
}

using json = nlohmann::json;using json = nlohmann::json;
bool log_json = true;
const std::string INSTRUCTION = "INSTRUCTION";
const std::string RELEASE = "RELEASE";
const std::string TIME = "TIME";
const std::string ARGS = "ARGS";
const std::string MEMORY = "MEMORY";
const std::string NAME = "NAME";
const std::string CONSTANT = "CONSTANT";

void DTRLogConstant(const std::string& name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = CONSTANT;
    j[NAME] = name;
    DTRLog(j.dump());
  } else {
    DTRLog(CONSTANT + " " + name);
  }
}

void DTRLogMemory(const std::string& name, size_t memory) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = MEMORY;
    j[NAME] = name;
    j[MEMORY] = std::to_string(memory);
    DTRLog(j.dump());
  } else {
    DTRLog(name + " " + MEMORY + ": " + std::to_string(memory));
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

}

void DTRLogCopy(const std::string& new_name, const std::string& old_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "COPY";
    j["DST"] = new_name;
    j["SRC"] = old_name;
    DTRLog(j.dump());
  } else {
    DTRLog(new_name + " = " + old_name);
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

std::tuple<Tensors, duration_t> make_raw(const rematerialize_function_t& remat,
                                         const strongs& input_values) {
  std::vector<Tensor> input;
  for (const strong& s: input_values) {
    CHECK(!s->t.key_set().has(DispatchKey::CheckpointTensorId));
    input.push_back(s->t);
  }
  time_t pre = std::chrono::system_clock::now();
  auto output = remat(input);
  time_t post = std::chrono::system_clock::now();
  return {output, post - pre};
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

void DTRLogCall(const std::vector<std::string>& res, const std::string& name, const std::vector<std::string>& args, const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "CALL";
    j[NAME] = name;
    j["RESULT"] = res;
    j[ARGS] = args;
    j[TIME] = time;
    DTRLog(j.dump());
  } else {
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
    DTRLog(log);
  }
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const Tensors& input) {
  strongs input_values;
  std::vector<std::string> args;
  for (const Tensor& t: input) {
    auto ft = from_tensor(t);
    input_values.push_back(std::get<0>(ft));
    args.push_back(std::get<1>(ft));
  }
  std::vector<std::string> res;
  auto ret = make_raw(remat, input_values);
  Tensors tensors;
  for (const Tensor& t: std::get<0>(ret)) {
    auto cp = checkpoint_raw(t);
    tensors.push_back(cp);
    res.push_back(get_cpti(cp)->counter_name());
  }
  DTRLogCall(res, name, args, from_time(std::get<1>(ret)));
  for (const Tensor& t: tensors) {
    auto cpti = get_cpti(t);
    DTRLogMemory(cpti->counter_name(), cpti->ref->value->memory());
  }
  return tensors;
}

void DTRLogMutate(const std::string& name, const std::vector<std::string>& args, const std::vector<size_t>& mutate, const std::string& time) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = "MUTATE";
    j[NAME] = name;
    j[ARGS] = args;
    j["MUTATE"] = mutate;
    j[TIME] = time;
    DTRLog(j.dump());
  } else {
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
    DTRLog(log);
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
  std::vector<std::string> args;
  for (const Tensor& t : inputs) {
    auto ft = from_tensor(t);
    args.push_back(std::get<1>(ft));
    input_values.push_back(std::get<0>(ft));
  }
  auto ret = make_raw(remat, input_values);
  const auto& modified = std::get<0>(ret);
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = intrusive_ptr<CheckpointTensorCell>::make(modified[idx]);
  }
  DTRLogMutate(name, args, mutate_idx, from_time(std::get<1>(ret)));
}

void DTRLogRelease(const std::string& counter_name) {
  if (log_json) {
    json j;
    j[INSTRUCTION] = RELEASE;
    j[NAME] = counter_name;
    DTRLog(j.dump());
  } else {
    DTRLog(RELEASE + ": " + counter_name);
  }
}

void CheckpointTensorImpl::release_resources() {
  DTRLogRelease(counter_name());
    ref.reset();
}

}
