#include "kernel_selection.h"
#include <stdexcept>
#include <string>

namespace prism {

constexpr WidthKernels kernelsFolded{kWidthFolded,
                                     launchEmbed,
                                     embedSharedBytes,
                                     nullptr,
                                     launchQkvFragments,
                                     launchAttention,
                                     launchFoldedCausalAttention,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     launchMlp,
                                     launchMlpKv,
                                     launchValueGraph,
                                     launchCandidate,
                                     launchKeyValue,
                                     launchReread,
                                     false};
constexpr WidthKernels kernelsProjected{kWidthProjected,
                                        nullptr,
                                        nullptr,
                                        launchWideEmbed,
                                        launchProjectedQkvFragments,
                                        launchProjectedAttention,
                                        launchProjectedCausalAttention,
                                        launchSharedProjection,
                                        launchPageAttention,
                                        launchPageMlp,
                                        launchMlp,
                                        launchMlpKv,
                                        launchValueGraph,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        true};

const WidthKernels& widthKernels(int hidden) {
  switch (hidden) {
#define PRISM_WIDTH(name, width, ...) \
  case width:                         \
    return kernels##name;
#define PRISM_RULES(seats, tiles)
#define PRISM_EAGER_WIDTH(name, width)
#include "collection_widths.def"
#undef PRISM_EAGER_WIDTH
#undef PRISM_RULES
#undef PRISM_WIDTH
    default:
      throw std::invalid_argument("no inference kernels for hidden width " + std::to_string(hidden));
  }
}

}  // namespace prism
