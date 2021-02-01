#ifndef THP_MODULE_INC
#define THP_MODULE_INC

#define THP_STATELESS_ATTRIBUTE_NAME "_torch"

#include <string>

struct CheckpointFunctions;
void SetCheckpointFunctions(CheckpointFunctions*);
CheckpointFunctions* GetCheckpointFunctions();

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

#endif
