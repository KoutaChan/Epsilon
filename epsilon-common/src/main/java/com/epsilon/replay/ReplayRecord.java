package com.epsilon.replay;

import java.util.List;

/** 読込済みの一対局。配列・イベントの所有権は読み込み処理から移譲し、利用側は変更しない。 牌山の配列は復元できた物理牌ID列だけを保持し、不明ならnullとする。 */
public record ReplayRecord(
    List<ReplayEvent> events, RecordMetadata metadata, int[][] walls, RecordCompletion completion) {

  /** 最終順位を教師に利用できる対局だけを返す。 */
  public ReplayRecord requireCompletedMatch() {
    if (completion != RecordCompletion.COMPLETE)
      throw new IllegalArgumentException("Training requires a completed match record.");
    if (!hasKnownFinalScores())
      throw new IllegalArgumentException(
          "Training requires recorded or fully reconstructible final scores.");
    return this;
  }

  /** 最終点が記録済み、または最終局の全精算を開始点から復元できるかを返す。 */
  public boolean hasKnownFinalScores() {
    if (completion != RecordCompletion.COMPLETE) return false;
    if (metadata.finalScores() != null) return true;
    boolean knownLedger = false;
    boolean settled = false;
    for (ReplayEvent event : events) {
      switch (event) {
        case ReplayEvent.StartKyoku start -> {
          knownLedger = start.scores() != null;
          settled = false;
        }
        case ReplayEvent.Hora win -> {
          knownLedger &= win.deltas() != null;
          settled = true;
        }
        case ReplayEvent.Ryukyoku draw -> {
          knownLedger &= draw.deltas() != null;
          settled = true;
        }
        case ReplayEvent.EndGame end -> {
          if (end.scores() != null) return true;
        }
        default -> {}
      }
    }
    return knownLedger && settled;
  }
}
