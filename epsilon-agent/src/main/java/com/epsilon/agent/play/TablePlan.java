package com.epsilon.agent.play;

import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Objects;

/**
 * 卓で打つ半荘の計画。牌山は run の種と牌山の番号だけで決まる。手の抽選は、局のキー（コホートの種、牌山の番号、局の序数から作る）と
 * 判断のキーだけで決まる（{@link TableSeeds}・{@link CrnKeys}）。割り当ての番号は乱数に入らない。
 *
 * @param runSeed 牌山の種の素
 * @param iterationSeed 局のキーの素
 * @param games 打つ半荘（卓の中で一意）。この順に始める
 * @param drawOrder 全ての半荘のツモの順
 * @param behaviour 全席の行動方策（null なら席の打ち手の分布から引く）
 * @param seating 半荘ごとの席の打ち手
 * @param observer 主経路の観測（null なら観測しない）
 * @param <Q> 問い合わせの型
 */
public record TablePlan<Q extends Query>(
    Variant variant,
    long runSeed,
    long iterationSeed,
    List<GameKey> games,
    DrawOrder drawOrder,
    Behaviour behaviour,
    Seating<Q> seating,
    GameObserver observer) {

  public TablePlan {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(games, "games");
    Objects.requireNonNull(drawOrder, "drawOrder");
    Objects.requireNonNull(seating, "seating");
  }

  /** 半荘ごとの席の打ち手。 */
  @FunctionalInterface
  public interface Seating<Q extends Query> {

    /**
     * 半荘の席の打ち手。半荘を始めるスレッドで、その半荘の観測より前に 1 度だけ呼ぶ。
     *
     * @return 席の順の打ち手（席の数だけ）。半荘の間は変えない
     */
    List<TableSeat<Q>> seats(GameKey game);
  }
}
