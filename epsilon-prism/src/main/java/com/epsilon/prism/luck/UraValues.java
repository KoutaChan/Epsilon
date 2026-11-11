package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.noise.UraSettings;
import java.util.List;
import java.util.Objects;

/**
 * 1 回の収集で、局末の値を求める。局末を作った進行のスレッドが、スレッドごとの作業領域（{@link UraExpectation}）で求め、局末の値の
 * 作り方に従って実際の値か期待値を書く。期待値は、期待値で学ぶときは全ての立直の和了で、実際の値で学ぶときは監査に選んだ局の立直の和了だけで求める。
 * どの局末も監査（{@link UraAudit}）に数える。
 *
 * <p>監査に選ぶかは、乱数 unit(derive(コホートの種, URA_AUDIT, 半荘, 局の序数)) で決める。乱数は局の結果にも牌山の中身にも依らず、同じ
 * 局の主経路・写し・分岐で同じなので、行の全ての手がそろって監査に入る。どのスレッドから呼んでもよい。
 */
public final class UraValues {

  private final UraSettings settings;
  private final long iterationSeed;
  private final ThreadLocal<UraExpectation> workspaces;
  private final UraAudit audit = new UraAudit();

  /**
   * @param settings 局末の値の作り方
   * @param phi 局末の卓の値を求める Φ
   * @param iterationSeed コホートの種
   */
  public UraValues(UraSettings settings, Phi phi, long iterationSeed) {
    this.settings = Objects.requireNonNull(settings, "settings");
    Objects.requireNonNull(phi, "phi");
    this.iterationSeed = iterationSeed;
    workspaces = ThreadLocal.withInitial(() -> new UraExpectation(phi));
  }

  /**
   * 局末の値を書き、局末を監査に数える。
   *
   * @param full 局末の局面
   * @param end 局の終わり方と結果
   * @param start 局の始まりの卓
   * @param history 終わった局までの各局の始まりの卓。Φ が履歴を読まなければ null でよい
   * @param gameId 半荘 ID
   * @param hand 半荘の中の局の序数
   * @param out 席ごとの局末の値の書き込み先
   * @return 値を求めた作業領域（このスレッドで次に呼ぶまで有効）
   */
  public UraExpectation handEnd(
      FullView full, RoundEnd end, TableState start, RoundHistory history, long gameId, int hand, double[] out) {
    UraExpectation values = workspaces.get();
    values.evaluate(full, end, start, history, settings.averages(coin(iterationSeed, gameId, hand)));
    values.values(settings, out);
    audit.handEnd(values);
    return values;
  }

  /**
   * 期待値を求めた局末なら、行の席の値を監査に数える。監査は行を局で束ねるので、{@link #handEnd} に渡した半荘と局を渡す。
   *
   * @param values {@link #handEnd} が返した作業領域
   * @param gameId 半荘 ID
   * @param hand 半荘の中の局の序数
   * @param rows 局末の値を受け取る行
   */
  public void rows(UraExpectation values, long gameId, int hand, List<? extends UraAudit.Row> rows) {
    audit.rows(values, gameId, hand, rows);
  }

  /** ここまでの監査の写し。 */
  public UraAudit.Report report() {
    return audit.report();
  }

  /** 局を監査に選ぶ乱数 unit(derive(コホートの種, URA_AUDIT, 半荘, 局の序数))。 */
  static double coin(long iterationSeed, long gameId, int hand) {
    return CrnSeeds.unit(CrnSeeds.derive(iterationSeed, CrnSeeds.Domain.URA_AUDIT, gameId, hand));
  }
}
