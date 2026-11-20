#pragma once

#include <cstddef>
#include <vector>

namespace prism::test {
enum class Command { kUpload, kDownload, kDeviceCopy, kClear, kLaunch, kAllocate, kHostWait, kEvent };

struct RecordedCommand {
  Command kind;
  const void* stream;
  const void* host;
  size_t bytes;
  int device;  // 記録した時に選ばれていたデバイス
};

void append(Command kind, const void* stream = nullptr, const void* host = nullptr, size_t bytes = 0);
const std::vector<RecordedCommand>& commands();
void resetCommands();
// 記録用の実装で選ばれているデバイス（gpu::setDevice と gpu::getDevice が読み書きする）。
void selectDevice(int device);
int selectedDevice();
void failNextCompletion();
bool completeStream(const void* stream);
// 次の count 回の確保は成功し、その後の確保は、負の数を渡し直すまで空きが足りずに失敗する。
void failAllocationAfter(int count);
bool allocationSucceeds();
// 記録したイベントを、次に問い合わせた時に未完了として返す。
void delayNextEvent();
bool eventComplete();
// 次のホストからデバイスへの転送を失敗させる。
void failNextUpload();
bool uploadSucceeds();
// デバイスの空きと総量（バイト）。
void setDeviceMemory(size_t available, size_t total);
size_t availableMemory();
size_t totalMemory();
}
