package com.epsilon.reviewer.engine;

import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.reviewer.dto.PreparedRecord;
import com.epsilon.reviewer.dto.RecordMetadata;
import java.util.Arrays;

/** 共通Readerの結果を一度再生し、推論開始前に記録行動の合法性を確認する。 */
final class RecordPreparation {
  private RecordPreparation() {}

  static PreparedRecord readPreparedRecord(byte[] bytes, String fileName) {
    ReplayRecord record = ReplayRecordReader.readRecord(bytes);
    int[] reconstructed;
    try {
      reconstructed = ReplayEngine.replay(record, (point, state, seat, legal) -> {});
    } catch (RuntimeException failure) {
      throw new IllegalArgumentException("Record replay failed: " + failure.getMessage(), failure);
    }
    var original = record.metadata();
    int[] finalScores = original.finalScores();
    String scoreSource = original.finalScoresSource();
    if (finalScores == null && record.hasKnownFinalScores()) {
      finalScores = reconstructed;
      scoreSource = original.format().sourceId() + ".terminal_deltas";
    }
    String name = fileName == null || fileName.isBlank() ? "record" : fileName;
    int rounds = (int) record.events().stream().filter(StartKyoku.class::isInstance).count();
    return new PreparedRecord(
        new RecordMetadata(
            original.format().sourceId(),
            name,
            original.names(),
            finalScores == null ? null : Arrays.stream(finalScores).boxed().toList(),
            scoreSource,
            rounds),
        record);
  }
}
