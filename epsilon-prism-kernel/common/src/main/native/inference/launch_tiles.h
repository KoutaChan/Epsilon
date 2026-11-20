#pragma once

namespace prism {

// 同時に計算する MLP の行数と FFN のチャンク数。広いタイルほど共有メモリとレジスタを多く使う。
struct MlpTuning {
  int rows;
  int chunks;
};

// ブロックに割り当てる行数と、読み込みを先行させる段数を定める。
struct DefaultKernelTuning {
  static constexpr bool kCanSplitShortPagedHeads = false;
  template <class Lane>
  static constexpr bool splitShortPagedHeads(const Lane&) {
    return false;
  }
  static constexpr MlpTuning kFoldedMlp{48, 1};
  static constexpr MlpTuning kProjectedMlp{32, 2};
  static constexpr int kPreferredCandidateRows = 48;
  static constexpr int kDictionaryRows = 32;
  static constexpr int kCandidateAhead = 3;
  static constexpr int kGateDepth = 12;
  static constexpr int kTokensPerBlock = 8;
  // 頻出辞書: 1 回に調べる組の数（辞書の 1 行あたり）、辞書を保った後と作り直した後に次に調べるまでの提出の数、辞書を使う命中率の
  // 下限（%）、作り直す命中率の低下と、作り直しに求める改善（ポイント）。
  static constexpr int kDictionarySamplesPerRow = 8;
  static constexpr int kDictionaryStableInterval = 32;
  static constexpr int kDictionaryChangeInterval = 8;
  static constexpr int kDictionaryMinimumCoverage = 75;
  static constexpr int kDictionaryCoverageDrop = 5;
  static constexpr int kDictionaryMinimumGain = 4;
};

}  // namespace prism
