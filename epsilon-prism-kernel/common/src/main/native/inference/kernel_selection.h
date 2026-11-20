#pragma once

#include "inference_stages.h"

namespace prism {

// 形を作るときに一度引き、起動時は選んだ起動関数だけを呼ぶ。
struct WidthKernels {
  WidthCapability capability;
  decltype(&launchEmbed) embed;
  decltype(&embedSharedBytes) embedBytes;
  decltype(&launchWideEmbed) projectedEmbed;
  decltype(&launchQkvFragments) qkv;
  decltype(&launchAttention) attention;
  decltype(&launchFoldedCausalAttention) causal;
  decltype(&launchSharedProjection) sharedProjection;
  decltype(&launchPageAttention) pageAttention;
  decltype(&launchPageMlp) pageMlp;
  decltype(&launchMlp) mlp;
  decltype(&launchMlpKv) lastMlp;
  decltype(&launchValueGraph) valueGraph;
  decltype(&launchCandidate) standard;
  decltype(&launchKeyValue) rereadKeys;
  decltype(&launchReread) reread;
  bool splitCandidate;  // 候補の Attention を前後の射影とは別に起動する。
};

const WidthKernels& widthKernels(int hidden);

}  // namespace prism
