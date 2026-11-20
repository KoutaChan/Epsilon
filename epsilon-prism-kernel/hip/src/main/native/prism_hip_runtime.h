// 読み込んだ HIP のランタイムと libtorch の情報（HipLibrary が DJL の側と照合する）。
#pragma once

#include <string>
#include <vector>

namespace prism {

// "名前=値" の並びを返す: hip.compiled・hip.runtime・hip.error（ランタイムのバージョンを取れなかったときだけ）・hip.library（この .so が
// リンクした libamdhip64）・hip.loaded（プロセスに読み込まれた全ての libamdhip64）・torch.loaded（libtorch*・libc10*）・cxx11abi・checked・
// kernels・regions・architecture（形の記述の int の数）・header（WeightPack のヘッダーの int の数）・arch。パスは実体のパスで、複数ある
// ときは ':' で区切る。
std::vector<std::string> runtimeInfo();

// 指定した GPU の arch（hipDeviceProp_t::gcnArchName）を返す。
std::string deviceArch(int device);

}  // namespace prism
