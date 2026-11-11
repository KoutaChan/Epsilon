package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import java.util.ArrayList;
import java.util.List;

/** 局の結果を作る。和了した席と放銃した席は牌譜の結果から、採点はエンジンの局末から、和了牌は表示用の卓の物理牌から取る。 */
public final class OutcomeProjector {
  private OutcomeProjector() {}

  /**
   * 和了 1 つ分の結果を作る。裏ドラの表示牌は、立直した和了にだけ付ける。
   *
   * @param table 局の最後のイベントまで進めた表示用の卓
   * @param result 牌譜の局の結果
   * @param end エンジンの局末（和了は結果と同じ順）
   * @param index 何番目の和了か
   */
  public static RoundOutcome.Hora win(
      RecordTable table, RoundResult result, RoundEnd end, int index) {
    int winner = result.wins().get(index).seat();
    int target = result.kind() == EndKind.TSUMO ? winner : result.loser();
    WinScore score = end.score(index);
    List<String> ura = new ArrayList<>(result.uraCount());
    if (score.has(ScoringYaku.RIICHI) || score.has(ScoringYaku.DOUBLE_RIICHI))
      for (int i = 0; i < result.uraCount(); i++)
        ura.add(TileCode.of(table.variant().tiles(), result.ura(i)));
    return new RoundOutcome.Hora(
        winner,
        target,
        table.winningTile(),
        WinDetails.of(table.variant(), table.round(), end.outcome(), index, score, ura));
  }

  /** 流局の結果を作る。 */
  public static RoundOutcome.Ryukyoku draw() {
    return new RoundOutcome.Ryukyoku();
  }
}
