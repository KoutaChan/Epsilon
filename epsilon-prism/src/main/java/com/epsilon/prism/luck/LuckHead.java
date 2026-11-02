package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import com.epsilon.prism.model.Context;
import com.epsilon.prism.model.Module;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;

/**
 * 運の網 Ω の結果の頭。牌のトークン t ごとに、その牌が結果になったときの局末の Φ を 4 席分出す。
 *
 * <pre>
 * a_t = SiLU(W_h·[T_t; G] + b_h)                 2H → G（計算の型）
 * u_t = SiLU(W_u·a_t + b_u)                      G → G（計算の型）
 * V(t, ρ)[r] = w[ρ, r]·u_t + b[ρ, r]             FP32、ρ ∈ {赤でない, 赤}、r = 行為者から見た相対席
 * V ← V − mean_r V                               4 席の和を 0 にする
 * </pre>
 *
 * <p>T_t・G は背骨の最後の正規化の後の牌・局のトークン。結果 37 通りは、34 種の赤でない牌の ρ = 赤でない と、正準の 3 色の 5 の ρ = 赤。結果 d は
 * トークンに入らず、どの結果の値を読むかにだけ使う（{@link ControlVariate}）。
 *
 * <p>Φ の 4 席の和は効用の和（定数）なので、4 席の和を 0 にした値が全ての情報を持つ。c は結果どうしの差しか読まないので定数に依らず、損失は 4 席
 * の和を 0 にした Y と比べる。最後の層は 0 で始めるので、白紙では V ≡ 0（c ≡ 0）。
 */
final class LuckHead extends Module {

  /** 結果の変種（赤でない・赤）。 */
  static final int VARIANTS = 2;

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int TILES = Tile.NUM_TILE_TYPES;

  private final Parameter hidden;
  private final Parameter hiddenBias;
  private final Parameter units;
  private final Parameter unitsBias;
  private final Parameter output;
  private final Parameter outputBias;

  /**
   * パラメーターを登録する。
   *
   * @param shape 網の形（頭の幅は G）
   */
  LuckHead(PrismShape shape) {
    int width = shape.gate();
    hidden =
        weight(
            "hidden",
            new Shape(width, 2L * shape.hidden()),
            Axis.of(Role.INNER_OUT),
            Axis.of(Role.STREAM_IN, 2));
    hiddenBias = bias("hiddenBias", width, Axis.of(Role.INNER_OUT));
    units =
        weight(
            "units", new Shape(width, width), Axis.of(Role.INNER_OUT), Axis.of(Role.INNER_IN));
    unitsBias = bias("unitsBias", width, Axis.of(Role.INNER_OUT));
    output =
        zeros(
            "output",
            new Shape((long) VARIANTS * SEATS, width),
            Axis.of(Role.FIXED),
            Axis.of(Role.INNER_IN));
    outputBias = zeros("outputBias", new Shape((long) VARIANTS * SEATS), Axis.of(Role.FIXED));
  }

  /**
   * 結果ごとの局末の Φ を出す。
   *
   * @param context 文脈
   * @param states 背骨の最後の正規化の後 [行, 39, H]（計算の型）
   * @return V [行, 37, 4]（FP32、4 席の和は 0。結果の軸は正準の結果の番号）
   */
  NDArray forward(Context context, NDArray states) {
    long rows = states.getShape().get(0);
    long width = states.getShape().get(2);
    int first = Token.TILE.first();
    NDArray tiles = states.get(new NDIndex(":, {}:{}, :", first, first + TILES));
    NDArray round =
        states
            .get(new NDIndex(":, {}:{}, :", Token.ROUND.first(), Token.ROUND.first() + 1))
            .broadcast(new Shape(rows, TILES, width));
    NDArray input = NDArrays.concat(new NDList(tiles, round), 2);
    NDArray activated = Context.silu(context.linear(input, hidden, hiddenBias));
    NDArray unit = Context.silu(context.linear(activated, units, unitsBias));
    NDArray values =
        context.linear(context.toFloat(unit), context.fp32(output), context.fp32(outputBias));
    NDList outcomes = new NDList(1 + LuckSuits.SUITS);
    outcomes.add(values.get(new NDIndex(":, :, 0:{}", SEATS)));
    for (int suit = 0; suit < LuckSuits.SUITS; suit++) {
      int five = suit * Tile.TILE_TYPES_PER_SUIT + Tile.M5;
      outcomes.add(values.get(new NDIndex(":, {}:{}, {}:", five, five + 1, SEATS)));
    }
    NDArray byOutcome = NDArrays.concat(outcomes, 1);
    return byOutcome.sub(byOutcome.mean(new int[] {2}, true));
  }
}
