package com.epsilon.arena;

import com.epsilon.runtime.MeasurementStatus;
import java.util.List;

/** 各参加枠を基準に集計した結果。順位差は正なら第0参加者が優勢。 */
public record ArenaResult(
    RunSettings settings,
    double seconds,
    double gamesPerSecond,
    List<PlayerResult> players,
    PairedResult paired,
    List<InferenceResult> inference) {
  public record PlayerResult(
      String name,
      double averageRank,
      double averageScore,
      int first,
      int second,
      int third,
      int fourth) {}

  public record PairedResult(
      int walls,
      double rankAdvantage,
      double rankStandardError,
      double rankLower95,
      double scoreAdvantage,
      double scoreStandardError,
      double scoreLower95) {}

  public record InferenceResult(
      String runtime,
      long rows,
      long batches,
      double rowsPerSecond,
      double batchFillRatio,
      double meanAdmissionQueueMicros,
      double meanRequestBatchMillis,
      ProviderTimings providerTimings) {}

  /** 各バッチのホスト経過時間。inferenceRoundTripはGPU キューと転送を含みカーネル時間ではない。 */
  public record ProviderTimings(
      MeasurementStatus status,
      long completedBatches,
      Double meanEncoderQueueMillis,
      Double meanEncodingMillis,
      Double meanInferenceRoundTripMillis) {}
}
