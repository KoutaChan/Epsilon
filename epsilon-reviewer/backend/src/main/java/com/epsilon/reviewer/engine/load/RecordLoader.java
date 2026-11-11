package com.epsilon.reviewer.engine.load;

import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.RecordFiles;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.ReplayDriver;
import java.util.Arrays;
import java.util.Locale;

/**
 * 牌譜のバイト列を読む。推論の前に全体を 1 回再生して、記録の手が合法か、局末が記録と合うかを確かめる。取り込むときも、保存した牌譜を
 * 読み直すときも同じ手順で読む。そのため、ルールが変わって再生できなくなった牌譜はここで見つかる。
 */
public final class RecordLoader {
  /** 牌譜を読むときに渡す id。Reviewer は牌譜を id で区別しないので、決まった値を渡す。 */
  private static final String RECORD_ID = "reviewer";

  private RecordLoader() {}

  /** @throws IllegalArgumentException 読めない、再生できない、または再生の結果が記録と食い違う */
  public static LoadedRecord load(byte[] bytes) {
    RecordedGame game = RecordFiles.read(bytes, RECORD_ID);
    try {
      new ReplayDriver(game.variant()).replay(game, new RuleWorkspace(game.variant()));
    } catch (RuntimeException failure) {
      throw new IllegalArgumentException("Record replay failed: " + failure.getMessage(), failure);
    }
    var summary =
        new RecordSummary(
            game.meta().source().name().toLowerCase(Locale.ROOT),
            game.meta().names(),
            Arrays.stream(game.finalScores()).boxed().toList(),
            game.rounds().size());
    return new LoadedRecord(summary, game);
  }
}
