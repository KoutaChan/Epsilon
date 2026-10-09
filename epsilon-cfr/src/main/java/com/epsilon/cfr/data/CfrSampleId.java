package com.epsilon.cfr.data;

import com.epsilon.util.SeedMixer;

/** 一つの学習区間で行を一意に識別する。ワーカーの処理完了順には依存しない。 */
public record CfrSampleId(long iteration, long episodeId, long decisionId, int seat)
    implements Comparable<CfrSampleId> {
  public CfrSampleId {
    if (iteration <= 0 || episodeId < 0 || decisionId < 0 || seat < 0 || seat >= 4) {
      throw new IllegalArgumentException("Invalid CFR sample identity");
    }
  }

  /** 内容や到着順に依存しない保持優先度。固定したシードから全世代で同じ値を求める。 */
  long retentionPriority(long seed) {
    long value = SeedMixer.mix64(seed ^ 0x6a09e667f3bcc909L);
    value = SeedMixer.mix64(value ^ iteration);
    value = SeedMixer.mix64(value ^ episodeId);
    value = SeedMixer.mix64(value ^ decisionId);
    return SeedMixer.mix64(value ^ seat);
  }

  @Override
  public int compareTo(CfrSampleId other) {
    int value = Long.compare(iteration, other.iteration);
    if (value == 0) value = Long.compare(episodeId, other.episodeId);
    if (value == 0) value = Long.compare(decisionId, other.decisionId);
    return value == 0 ? Integer.compare(seat, other.seat) : value;
  }
}
