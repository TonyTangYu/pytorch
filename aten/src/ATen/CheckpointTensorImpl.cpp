#include <ATen/CheckpointTensorImpl.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {

namespace native {

Tensor checkpoint(const Tensor& t) {
  throw;
}

Tensor uncheckpoint(const Tensor& t) {
  throw;
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
