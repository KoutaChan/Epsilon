package com.epsilon.prism;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.prism.core.Threats;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * テスト用のヘルパー。エンジンで打った判断ごとに、観測（O）と、同じ判断を卓の席から見た値の組を集める。卓の席から見た値は、収集が卓の側で
 * 決める値。
 */
public final class ObservedDecisions {

  private ObservedDecisions() {}

  /**
   * 1 つの判断。
   *
   * @param words 観測のワードのコピー
   * @param legal 合法手の行動の番号（エンジンの判断に並んだ順）
   * @param kind エンジンの判断の種類
   * @param threats 席から公開情報で見た脅威の数
   * @param round 局の番号
   * @param liveLeft 卓の牌山の残り枚数
   */
  public record Decision(
      long[] words, int[] legal, EngineDecisionKind kind, int threats, int round, int liveLeft) {

    /** 観測を、新しく作ったリーダーで読む。 */
    public Observation observation() {
      return new Observation(TENHOU_4P).bind(words, 0);
    }
  }

  /** 乱数の種 {@code seed} から半荘を {@code games} 回打ち、合法手が 2 つ以上ある判断を {@code limit} 個まで集める。 */
  public static List<Decision> collect(long seed, int games, TableWalk.Policy policy, int limit) {
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    List<Decision> out = new ArrayList<>();
    new TableWalk(TENHOU_4P, policy)
        .run(
            seed,
            games,
            (engine, slot) -> {
              Decisions decisions = engine.decisions();
              int n = decisions.legalCount(slot);
              if (out.size() >= limit || n < 2) return;
              SeatView view = engine.view(decisions.seat(slot));
              batch.clear();
              int index = writer.capture(view, decisions, slot, batch);
              int start = batch.start(index);
              int[] legal = new int[n];
              for (int a = 0; a < n; a++) legal[a] = decisions.legal(slot, a);
              out.add(
                  new Decision(
                      Arrays.copyOfRange(batch.words(), start, start + batch.longs(index)),
                      legal,
                      decisions.kind(slot),
                      Threats.count(view),
                      view.round(),
                      engine.full().seat(0).liveLeft()));
            });
    return out;
  }
}
