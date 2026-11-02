#include "prism_hip_runtime.h"

#include <dlfcn.h>
#include <limits.h>
#include <link.h>
#include <stdlib.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "prism_hip_common.h"
#include "prism_hip_plan.h"

namespace prism {
namespace {

std::string realPath(const char* path) {
  char resolved[PATH_MAX];
  return realpath(path, resolved) != nullptr ? std::string(resolved) : std::string(path);
}

std::string baseName(const std::string& path) {
  const size_t slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool startsWith(const std::string& text, const char* prefix) {
  return text.compare(0, std::strlen(prefix), prefix) == 0;
}

// プロセスに読み込まれた共有の物の実のパス（重なりは除く）。
std::vector<std::string> loadedObjects() {
  std::vector<std::string> objects;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') {
          static_cast<std::vector<std::string>*>(data)->push_back(realPath(info->dlpi_name));
        }
        return 0;
      },
      &objects);
  std::sort(objects.begin(), objects.end());
  objects.erase(std::unique(objects.begin(), objects.end()), objects.end());
  return objects;
}

std::string join(const std::vector<std::string>& values) {
  std::string joined;
  for (const std::string& value : values) {
    if (!joined.empty()) {
      joined += ':';
    }
    joined += value;
  }
  return joined;
}

}  // namespace

std::vector<std::string> runtimeInfo() {
  std::vector<std::string> info;
  info.push_back("hip.compiled=" + std::to_string(HIP_VERSION));
  int runtime = -1;
  const hipError_t status = hipRuntimeGetVersion(&runtime);
  info.push_back("hip.runtime=" + std::to_string(status == hipSuccess ? runtime : -1));
  if (status != hipSuccess) {
    info.push_back(std::string("hip.error=") + hipGetErrorString(status));
  }

  Dl_info symbol{};
  if (dladdr(reinterpret_cast<void*>(&hipRuntimeGetVersion), &symbol) == 0 || symbol.dli_fname == nullptr) {
    throw std::runtime_error("cannot locate the HIP runtime that libprism_hip.so is bound to");
  }
  info.push_back("hip.library=" + realPath(symbol.dli_fname));

  std::vector<std::string> hip;
  std::vector<std::string> torch;
  for (const std::string& object : loadedObjects()) {
    const std::string name = baseName(object);
    if (startsWith(name, "libamdhip64")) {
      hip.push_back(object);
    } else if (startsWith(name, "libtorch") || startsWith(name, "libc10")) {
      torch.push_back(object);
    }
  }
  info.push_back("hip.loaded=" + join(hip));
  info.push_back("torch.loaded=" + join(torch));
  info.push_back("cxx11abi=" + std::to_string(_GLIBCXX_USE_CXX11_ABI));
  info.push_back(std::string("checked=") + (PRISM_HIP_CHECKED ? "true" : "false"));
  info.push_back("kernels=" + std::to_string(kKernels));
  info.push_back("regions=" + std::to_string(kRegions));
  info.push_back("header=" + std::to_string(kHeaderInts));
  info.push_back("arch=" PRISM_HIP_ARCH);
  return info;
}

std::string deviceArch(int device) {
  hipDeviceProp_t properties;
  PRISM_HIP(hipGetDeviceProperties(&properties, device));
  return properties.gcnArchName;
}

}  // namespace prism
