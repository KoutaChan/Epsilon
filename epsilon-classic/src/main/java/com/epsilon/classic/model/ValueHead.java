package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import com.epsilon.classic.input.ClassicFields.BoundaryFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.Tensor;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;

/**
 * 価値ヘッド。価値に使う局面の表現から 101 個のビンごとの残差を作り、その平均を引いてから、GRP の事前分布から作った分布の対数に足す。
 *
 * <p>事前分布は次のように作る。局の始まりの GRP から自分の絶対席の順位の確率を取り、その確率で順位の効用を平均する。この効用を、
 * 教師と同じ HL-Gauss の分布に変え、確率を {@link HlGauss#PRIOR_FLOOR} 以上にしてから対数を取る。勾配は流さない。GRP の事前分布が
 * 無い行では、順位の分布が一様になる。残差の射影は 0 で初期化するので、学習の始めの価値は事前分布そのものになる。
 *
 * <p>層（価値の隠れ層、Major だけにある残差の層、ビンの層）は、ネットワークが直下の子として持つ。このクラスが持つのは計算と定数の表だけ。
 */
final class ValueHead {

  private static final int SEATS = ClassicSchema.SEATS;

  private final Linear hiddenHead;
  private final Linear residualHead;
  private final Linear outputHead;
  private final HlGauss support;
  private final float[] utilities;
  private final int playerSeat;
  private final int grpRank;

  /** 絶対席の番号 0〜3 の表 [1, 4]（INT32）と、順位ごとの効用の表 [順位]（FP32）。 */
  private NDArray seatIds;

  private NDArray utilityValues;

  /**
   * @param residualHead 隠れ層の残差の層（無ければ null）
   */
  ValueHead(
      ClassicSchema schema,
      Variant variant,
      RankUtility utility,
      Linear hiddenHead,
      Linear residualHead,
      Linear outputHead) {
    this.hiddenHead = hiddenHead;
    this.residualHead = residualHead;
    this.outputHead = outputHead;
    support = new HlGauss(variant, utility);
    utilities = new float[SEATS];
    for (int rank = 0; rank < utilities.length; rank++) {
      utilities[rank] = (float) utility.value(variant, rank);
    }
    playerSeat = schema.layout(RoundInt.class).offset(RoundInt.PLAYER_SEAT);
    grpRank = schema.layout(BoundaryFloat.class).offset(BoundaryFloat.GRP_RANK);
  }

  HlGauss support() {
    return support;
  }

  /**
   * 定数の表を、順伝播のたびに作らずに済むよう先に作る。ネットワークの初期化で 1 回呼ぶ。
   *
   * @param manager パラメーターと定数を持つマネージャー（デバイスもこのマネージャーのもの）
   */
  void initialize(NDManager manager) {
    int[] seats = new int[SEATS];
    for (int seat = 0; seat < SEATS; seat++) {
      seats[seat] = seat;
    }
    seatIds = manager.create(seats, new Shape(1, SEATS));
    utilityValues = manager.create(utilities);
    support.initialize(manager);
  }

  /**
   * @param state 価値に使う局面の表現 [行, 幅]
   * @param batch 行のバッチ（自分の席と GRP の事前分布を読む）
   * @return ビンのロジット [行, 101]（FP32）
   */
  NDArray logits(Pass pass, NDArray state, ClassicBatch batch) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray hidden = Pass.silu(pass.linear(hiddenHead, state));
      if (residualHead != null) {
        hidden = hidden.add(pass.linear(residualHead, Pass.silu(hidden)));
      }
      NDArray residual = pass.toFloat(pass.linear(outputHead, hidden));
      return scope.keep(prior(batch).add(residual.sub(residual.mean(new int[] {1}, true))));
    }
  }

  /** GRP の事前分布から作った分布の対数 [行, 101]（FP32）。勾配は流さない。 */
  private NDArray prior(ClassicBatch batch) {
    int rows = batch.rows();
    NDArray marginals =
        batch
            .tensor(Tensor.BOUNDARY_CONTEXT)
            .get(":, {}:{}", grpRank, grpRank + BoundaryFloat.GRP_RANK.count())
            .stopGradient()
            .reshape(rows, SEATS, SEATS);
    NDArray seat =
        batch.tensor(Tensor.STATE_CATEGORIES).get(":, {}", playerSeat).toType(DataType.INT32, false).sub(1);
    NDArray own = seat.reshape(rows, 1).eq(seatIds).toType(DataType.FLOAT32, false).reshape(rows, SEATS, 1);
    NDArray ranks = marginals.mul(own).sum(new int[] {1});
    NDArray utility = ranks.mul(utilityValues).sum(new int[] {1});
    return support.targets(utility).maximum(HlGauss.PRIOR_FLOOR).log().stopGradient();
  }
}
