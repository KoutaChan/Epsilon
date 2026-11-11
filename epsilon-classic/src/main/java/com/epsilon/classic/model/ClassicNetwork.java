package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * Nano と Major のネットワーク。合法手ごとの方策のスコアと、効用の 101 ビンの分布（価値）を返す。
 *
 * <p>方策と価値は状態のエンコーダー（{@link StateEncoder}）を共有し、集約（{@link Readout}）はそれぞれが持つ。価値の集約は勾配を止めた
 * 表現を読むので、価値の損失は共有のエンコーダーを動かさない。
 *
 * <p>子の層の名前と登録の順は保存の形式（weights.bin の並び）の一部で、変換した重みもこれに合わせてある。順に、状態のエンコーダー、方策の集約、
 * 価値の集約、方策ヘッド、価値の隠れ層、価値の残差の層（Major だけ）、価値のビンの層。パラメーターは FP32 で持ち、計算の型は順伝播ごとに
 * 選ぶ（CPU では FP32、GPU では BF16）。
 */
public final class ClassicNetwork extends AbstractBlock {

  /** 順伝播で求めるヘッド。 */
  public enum Heads {
    POLICY(true, false),
    BOTH(true, true);

    private final boolean policy;
    private final boolean value;

    Heads(boolean policy, boolean value) {
      this.policy = policy;
      this.value = value;
    }

    public boolean policy() {
      return policy;
    }

    public boolean value() {
      return value;
    }
  }

  /**
   * 順伝播の出力。
   *
   * @param policy 方策のスコア（求めなければ null）
   * @param value 価値のビンのロジット [行, 101]（FP32、求めなければ null）
   */
  public record Output(PolicyScores policy, NDArray value) {}

  /** パラメーターの名前の先頭に付く、子の番号の桁数。 */
  private static final int CHILD_NUMBER = 2;

  private static final String VALUE_CHILD = "value";

  private final Architecture architecture;
  private final Variant variant;
  private final RankUtility utility;
  private final ClassicSchema schema;
  private final int hidden;
  private final StateEncoder stateEncoder;
  private final Readout policyStateReadout;
  private final Readout valueStateReadout;
  private final PolicyHead policyHead;
  private final Linear valueHiddenHead;
  private final Linear valueResidualHead;
  private final Linear valueHead;
  private final ValueHead value;

  /**
   * @param architecture アーキテクチャ
   * @param variant ルール（四人打ちだけ）
   * @param utility 価値の効用。Major では、点数の計算で求める終局の効用にも使う
   */
  public ClassicNetwork(Architecture architecture, Variant variant, RankUtility utility) {
    this.architecture = Objects.requireNonNull(architecture, "architecture");
    this.variant = Objects.requireNonNull(variant, "variant");
    this.utility = Objects.requireNonNull(utility, "utility");
    if (variant.seats() != ClassicSchema.SEATS) {
      throw new IllegalArgumentException("Classic networks need " + ClassicSchema.SEATS + " seats, not " + variant);
    }
    schema = ClassicSchema.of(architecture);
    Widths widths = Widths.of(architecture);
    hidden = widths.hidden();
    stateEncoder = addChildBlock("stateEncoder", new StateEncoder(widths));
    policyStateReadout =
        addChildBlock("policyStateReadout", new Readout(widths, Widths.POLICY_READOUT_FEED_FORWARD));
    valueStateReadout = addChildBlock("valueStateReadout", new Readout(widths, Widths.VALUE_READOUT_FEED_FORWARD));
    policyHead = addChildBlock("policyHead", new PolicyHead(widths, variant, utility));
    valueHiddenHead = addChildBlock("valueHiddenHead", Linear.builder().setUnits(hidden).build());
    valueResidualHead =
        architecture == Architecture.MAJOR ? addChildBlock("valueResidualHead", zeroLinear(hidden)) : null;
    valueHead = addChildBlock("valueHead", zeroLinear(HlGauss.BINS));
    value = new ValueHead(schema, variant, utility, valueHiddenHead, valueResidualHead, valueHead);
  }

  public Architecture architecture() {
    return architecture;
  }

  public Variant variant() {
    return variant;
  }

  public RankUtility utility() {
    return utility;
  }

  /** 価値のビン。 */
  public HlGauss support() {
    return value.support();
  }

  /**
   * 価値の損失だけが勾配を流すパラメーターなら true を返す。価値の層は、ネットワークの直下の子のうち名前が value で始まるもの（価値の集約・
   * 隠れ層・残差の層・ビンの層）。パラメーターの名前は、子の番号（2 桁）と子の名前で始まる。ほかのパラメーターには、方策の損失だけが勾配を流す。
   *
   * @param name {@code getParameters()} のパラメーターの名前
   */
  public static boolean valueParameter(String name) {
    return name.length() > CHILD_NUMBER && name.startsWith(VALUE_CHILD, CHILD_NUMBER);
  }

  /**
   * チェックポイントに書くヘッダーを返す。
   *
   * @param step 学習のステップ数
   */
  public ClassicHeader header(long step) {
    return new ClassicHeader(architecture, variant, utility, schema.fingerprint(), step);
  }

  /**
   * パラメーターを FP32 で作る。値は乱数で、0 で始める層だけは 0 にする。定数の表もここで作るので、重みを読む前にも呼ぶ。
   *
   * @param manager パラメーターと定数を持つマネージャー（デバイスもこのマネージャーのもの）
   */
  public void initialize(NDManager manager) {
    initialize(manager, DataType.FLOAT32, ClassicBatch.initializationShapes(schema));
  }

  /**
   * 順伝播を計算する。出力の配列は、行のバッチのマネージャーに付く。途中の配列は、次の部分が読む値を作り終えた時点で閉じる
   * （{@link ArrayScope}）。
   *
   * @param store パラメーターの値を返すストア
   * @param batch 行のバッチ（このネットワークと同じアーキテクチャのもの）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   * @param heads 求めるヘッド
   * @return 出力
   */
  public Output forward(
      ParameterStore store, ClassicBatch batch, Precision precision, boolean training, Heads heads) {
    if (batch.schema() != schema) {
      throw new IllegalArgumentException(
          "A " + architecture + " network cannot read " + batch.schema().architecture() + " inputs");
    }
    Pass pass = new Pass(store, batch.manager(), precision, training);
    try (ArrayScope scope = new ArrayScope()) {
      Memory memory = stateEncoder.encode(pass, batch);
      PolicyScores policy =
          heads.policy() ? policyHead.score(pass, batch, memory, policyStateReadout.read(pass, memory)) : null;
      NDArray values =
          heads.value() ? value.logits(pass, valueStateReadout.read(pass, memory.detached()), batch) : null;
      if (policy != null) {
        scope.keep(policy.alternatives());
        scope.keep(policy.actions());
        scope.keep(policy.riichi());
      }
      return new Output(policy, scope.keep(values));
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    stateEncoder.initialize(manager, dataType, inputShapes);
    Shape entities = new Shape(-1, StateEncoder.ENTITIES, hidden);
    policyStateReadout.initialize(manager, dataType, entities);
    valueStateReadout.initialize(manager, dataType, entities);
    policyHead.initialize(manager, dataType, inputShapes);
    Shape state = new Shape(-1, hidden);
    valueHiddenHead.initialize(manager, dataType, state);
    if (valueResidualHead != null) {
      valueResidualHead.initialize(manager, dataType, state);
    }
    valueHead.initialize(manager, dataType, state);
    value.initialize(manager);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use forward with a ClassicBatch");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {
      new Shape(-1, Alternative.values().length), new Shape(-1, -1), new Shape(-1, -1), new Shape(-1, HlGauss.BINS)
    };
  }

  /** 重みとバイアスを 0 で始める線形の層。 */
  private static Linear zeroLinear(int units) {
    Linear block = Linear.builder().setUnits(units).build();
    block.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0f));
    block.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0f));
    return block;
  }
}
