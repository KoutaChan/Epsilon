package com.epsilon.reviewer.engine.load;

import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.reviewer.engine.view.EventKind;
import java.util.List;

/**
 * 読み込んだ牌譜と、解析結果の文書での手順の位置を引くインデックス。手順は局ごとに、start_kyoku、牌譜のイベント（立直の宣言牌の打牌の前には
 * reach が入る）、hora（和了ごとに 1 つ）か ryukyoku、end_kyoku の順に並ぶ。最後の局の後に end_game が来る。位置は文書の全体を通した 0 からの
 * 番号。インデックスは読み込むときに 1 回だけ作り、その後は変えない。
 */
public final class LoadedRecord {
  private final RecordSummary summary;
  private final RecordedGame game;

  /** [局][牌譜のイベントの位置] → そのイベントの最初の手順の位置。イベントの位置にイベントの数を使うと、局の結果の最初の手順の位置になる。 */
  private final int[][] steps;

  /** 手順の数（end_game を含む）。 */
  private final int stepCount;

  LoadedRecord(RecordSummary summary, RecordedGame game) {
    this.summary = summary;
    this.game = game;
    List<RecordedRound> rounds = game.rounds();
    steps = new int[rounds.size()][];
    int step = 0;
    for (int round = 0; round < rounds.size(); round++) {
      RecordedRound recorded = rounds.get(round);
      List<RecordEvent> events = recorded.events();
      int[] positions = new int[events.size() + 1];
      step++;
      for (int index = 0; index < events.size(); index++) {
        positions[index] = step;
        step += EventKind.declaresRiichi(events.get(index)) ? 2 : 1;
      }
      positions[events.size()] = step;
      step += outcomeSteps(recorded) + 1;
      steps[round] = positions;
    }
    stepCount = step + 1;
  }

  public RecordSummary summary() {
    return summary;
  }

  public RecordedGame game() {
    return game;
  }

  public List<RecordedRound> rounds() {
    return game.rounds();
  }

  /** 局 {@code round} の start_kyoku の手順の位置。 */
  public int firstStep(int round) {
    return steps[round][0] - 1;
  }

  /**
   * 局 {@code round} の、牌譜のイベント {@code index} の最初の手順の位置。
   *
   * @param index 牌譜のイベントの位置。イベントの数を渡すと、局の結果（hora か ryukyoku）の最初の手順を返す
   */
  public int step(int round, int index) {
    return steps[round][index];
  }

  /** 文書の手順の数。 */
  public int stepCount() {
    return stepCount;
  }

  /** 局の結果の手順の数。和了なら和了ごとに hora が 1 つ、それ以外は ryukyoku が 1 つ。 */
  public static int outcomeSteps(RecordedRound round) {
    return round.result().win() ? round.result().wins().size() : 1;
  }
}
