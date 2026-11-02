// 読み込まれた HIP の実行系と libtorch の事実（HipLibrary が DJL と照らす）。
#pragma once

#include <string>
#include <vector>

namespace prism {

// "名前=値" の列: hip.compiled・hip.runtime・hip.library（この .so が繋がった libamdhip64）・hip.loaded（プロセスにある全ての
// libamdhip64）・torch.loaded（libtorch*・libc10*）・cxx11abi・checked・kernels・regions・header（WeightPack の頭の int の数）・arch。パスは実のパスで、列は ':' で区切る。
std::vector<std::string> runtimeInfo();

// GPU device の arch（hipDeviceProp_t::gcnArchName）。
std::string deviceArch(int device);

}  // namespace prism
