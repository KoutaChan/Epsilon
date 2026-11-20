#include "inference_device.h"

namespace prism::gpu {
Status getDevice(int* value) {
  *value = test::selectedDevice();
  return success;
}
Status setDevice(int value) {
  test::selectDevice(value);
  return success;
}
Status memoryInfo(size_t* available, size_t* total) {
  *available = test::availableMemory();
  *total = test::totalMemory();
  return success;
}
Status copy(void* destination, const void* source, size_t bytes, Direction direction, Stream stream) {
  using test::Command;
  switch (direction) {
    case hostToDevice:
      if (!test::uploadSucceeds()) return failed;
      test::append(Command::kUpload, stream, source, bytes);
      break;
    case deviceToHost:
      test::append(Command::kDownload, stream, destination, bytes);
      break;
    case deviceToDevice:
      test::append(Command::kDeviceCopy, stream, nullptr, bytes);
      break;
  }
  return success;
}
Status clear(void*, int, size_t bytes, Stream stream) {
  test::append(test::Command::kClear, stream, nullptr, bytes);
  return success;
}
Status synchronize(Stream stream) { return test::completeStream(stream) ? success : failed; }
Status createEventFlags(Event* event, unsigned) {
  test::append(test::Command::kAllocate);
  *event = new int{};
  return success;
}
Status destroyEvent(Event event) {
  delete event;
  return success;
}
Status record(Event, Stream stream) {
  test::append(test::Command::kEvent, stream);
  return success;
}
}
