#include <ATen/CheckpointTensorImpl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {

CheckpointTensorImpl* get_cpti(const Tensor& t) {
  return dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
}

CheckpointTensorImpl* must_get_cpti(const Tensor& t) {
  auto ret = get_cpti(t);
  TORCH_CHECK(ret);
  return ret;
}

namespace native {

Tensor checkpoint(const Tensor& t) {
  TORCH_CHECK(!is_checkpoint(t));
  auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t);
  return Tensor(cpti);
}

Tensor uncheckpoint(const Tensor& t) {
  auto cpti = must_get_cpti(t);
  return cpti->t;
}

void pin(Tensor& t) {
  throw;
}

Tensor try_uncheckpoint(const Tensor& t) {
  return is_checkpoint(t) ? uncheckpoint(t) : t;
}

Tensor decheckpoint(const Tensor& t) {
  return try_uncheckpoint(t);
}

bool is_checkpoint(const Tensor& t) {
  return get_cpti(t) != nullptr;
}

Tensor try_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? t : checkpoint(t);
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

  for (size_t i = 0; i < num_arg; ++i) {
    reversed_in.push_back(torch::jit::pop(stack));
  }
  // todo: is it safe to save a torch::jit::stack*?
  // todo: modify on heap instead of pushing and popping?
  auto call = [=](){
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
