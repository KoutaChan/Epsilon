package com.epsilon.reviewer.engine;

import com.epsilon.replay.ReplayRecord;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import java.util.List;

/**
 * もしもの試験の牌譜。epsilon-common の試験が持つ天鳳の鳳凰卓の牌譜（ダブロン・加槓・槓ドラ・リーチを含み、記録の牌山を持つ）を、試験の
 * resources として読む（pom の testResources）。
 */
final class GoldenRecords {
  static final List<String> NAMES = List.of("double-ron-a", "double-ron-b", "kakan-a", "kakan-b");

  private GoldenRecords() {}

  static LoadedRecord read(String name) {
    return RecordLoader.load(TestFiles.golden(name));
  }

  /** 記録の牌山を外した同じ牌譜（mjai・雀魂と同じく、ドラ表示を局の始めの出来事から読む再生）。 */
  static LoadedRecord withoutWalls(LoadedRecord record) {
    ReplayRecord replay = record.replay();
    return new LoadedRecord(
        record.summary(),
        new ReplayRecord(replay.events(), replay.metadata(), null, replay.completion()),
        record.roundStarts());
  }
}
