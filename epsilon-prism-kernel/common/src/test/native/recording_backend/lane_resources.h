#pragma once

#include "inference_device.h"
#include <cassert>

namespace prism::gpu {
class SelectedDevice {
 public:
  explicit SelectedDevice(int device) {
    getDevice(&previous_);
    setDevice(device);
  }
  ~SelectedDevice() { setDevice(previous_); }

 private:
  int previous_;
};

class Lane {
 public:
  Lane(int, Stream stream) { assert(stream); }
  size_t sharedBytes() const { return 65536; }
};
}

namespace prism {
inline bool eventComplete(gpu::Event) { return test::eventComplete(); }
}
