package com.epsilon.reviewer.engine.load;

import com.epsilon.replay.RecordMetadata;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayRecord;
import com.epsilon.replay.ReplayRecordReader;
import java.util.Arrays;

/** 牌譜のバイト列を共通の読み手で読む。取り込みは確かめの再生で記録の手の合法性を推論の前に確かめる。 */
public final class RecordLoader {
  private RecordLoader() {}

  /**
   * 読み、全体を 1 回再生して確かめる。最終点が牌譜に無ければ、全ての精算から復元できる時だけ再生の点を使う。
   *
   * @throws IllegalArgumentException 読めない・再生できない（文は英語）
   */
  public static LoadedRecord load(byte[] bytes) {
    ReplayRecord replay = ReplayRecordReader.readRecord(bytes);
    int[] reconstructed;
    try {
      reconstructed = ReplayEngine.replay(replay, (decision, state, legal) -> {});
    } catch (RuntimeException failure) {
      throw new IllegalArgumentException("Record replay failed: " + failure.getMessage(), failure);
    }
    RecordMetadata metadata = replay.metadata();
    int[] finalScores = metadata.finalScores();
    if (finalScores == null && replay.hasKnownFinalScores()) finalScores = reconstructed;
    int[] starts = LoadedRecord.roundStarts(replay.events());
    var summary =
        new RecordSummary(
            metadata.format().sourceId(),
            metadata.names(),
            finalScores == null ? null : Arrays.stream(finalScores).boxed().toList(),
            starts.length);
    return new LoadedRecord(summary, replay, starts);
  }

  /**
   * 保存した牌譜を確かめの再生なしで読み直す（取り込みの時に確かめた）。
   *
   * @param summary 保存の時の概要
   */
  public static LoadedRecord reload(byte[] bytes, RecordSummary summary) {
    ReplayRecord replay = ReplayRecordReader.readRecord(bytes);
    return new LoadedRecord(summary, replay, LoadedRecord.roundStarts(replay.events()));
  }
}
