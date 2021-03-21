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

Tensor decheckpoint(const Tensor& t) {
  throw;
}

bool is_checkpoint(const Tensor& t) {
  throw;
}


Tensor try_checkpoint(const Tensor& t) {
  throw;
}

}

void CheckpointFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "inside fallback" << std::endl;
  std::cout << op.operator_name() << std::endl;
  auto s = op.schema();
  size_t num_arg = s.arguments().size(), num_ret = s.returns().size();
  std::cout << num_arg << std::endl;
  std::cout << num_ret << std::endl;
  TORCH_CHECK(!s.is_mutable()); // todo: deal with mutability. s.hasAnyAliasInfo() might be useful.
  std::vector<IValue> reversed_in, reversed_out; // popping them from the jit stack and pushing them back will reverse stuff.
  for (size_t i = 0; i < num_arg; ++i) {
    reversed_in.push_back(torch::jit::pop(stack));
  }
  // rematerializer: grab the reversed_in, uncheckpoint and push all of them onto the stack,
  // grab the output arguments from the stack and push them into reversed_out,
  // then walk through all the tensor to update a list of receivers.
  auto ivalue_uncheckpoint() = [](){};
  auto ivalue_checkpoint() = [](){};
  for (size_t i = 0; i < num_ret; ++i) {
    reversed_out.push_back(torch::jit::pop(stack));
  }
  op.callBoxed(stack);
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
