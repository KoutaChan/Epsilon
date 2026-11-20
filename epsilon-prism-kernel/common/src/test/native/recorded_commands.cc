#include "recorded_commands.h"

#include <utility>

namespace prism::test {
namespace {
std::vector<RecordedCommand> history;
bool failCompletion = false;
int allocationsBeforeFailure = -1;
bool delayEvent = false;
bool failUpload = false;
size_t deviceAvailable = size_t{4} << 30;
size_t deviceTotal = size_t{4} << 30;
int device = 0;
}

void append(Command kind, const void* stream, const void* host, size_t bytes) {
  history.push_back({kind, stream, host, bytes, device});
}

const std::vector<RecordedCommand>& commands() { return history; }
void resetCommands() { history.clear(); }
void selectDevice(int value) { device = value; }
int selectedDevice() { return device; }
void failNextCompletion() { failCompletion = true; }

bool completeStream(const void* stream) {
  append(Command::kHostWait, stream);
  return !std::exchange(failCompletion, false);
}

void failAllocationAfter(int count) { allocationsBeforeFailure = count; }

bool allocationSucceeds() {
  if (allocationsBeforeFailure < 0) return true;
  if (allocationsBeforeFailure == 0) return false;
  --allocationsBeforeFailure;
  return true;
}

void delayNextEvent() { delayEvent = true; }

bool eventComplete() { return !std::exchange(delayEvent, false); }

void failNextUpload() { failUpload = true; }

bool uploadSucceeds() { return !std::exchange(failUpload, false); }

void setDeviceMemory(size_t available, size_t total) {
  deviceAvailable = available;
  deviceTotal = total;
}

size_t availableMemory() { return deviceAvailable; }

size_t totalMemory() { return deviceTotal; }
}
