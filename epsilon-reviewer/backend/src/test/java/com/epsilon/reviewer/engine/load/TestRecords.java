package com.epsilon.reviewer.engine.load;

import com.epsilon.mahjong.record.RecordedGame;
import java.util.Arrays;
import java.util.Locale;

/** テスト用の牌譜を、再生で確かめずに読み込んだ形にする。手で作った局は途中で終わるので、再生では確かめられない。 */
public final class TestRecords {
  private TestRecords() {}

  public static LoadedRecord unchecked(RecordedGame game) {
    var summary =
        new RecordSummary(
            game.meta().source().name().toLowerCase(Locale.ROOT),
            game.meta().names(),
            Arrays.stream(game.finalScores()).boxed().toList(),
            game.rounds().size());
    return new LoadedRecord(summary, game);
  }
}
