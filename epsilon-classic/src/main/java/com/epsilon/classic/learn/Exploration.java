package com.epsilon.classic.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.settings.Key;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.VocabularyCodec;
import com.epsilon.classic.model.PolicyGraph;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import java.util.Objects;

/**
 * 自己対局で手を選ぶときの振る舞いの方策 μ。方策のグラフ（{@link PolicyGraph}）の節点ごとに、方策の条件つき分布へ候補の一様分布を
 * 割合 ε だけ混ぜる。手の確率は、その手までに通った節点の確率の積。最後に、全ての合法手の確率を下限 f 以上にする。
 *
 * <pre>
 * 節点: μ(x | 節点) = (1 − ε)·π(x | 節点) + ε / 候補の数     （片側しか無い二択は分けない）
 * 下限: μ(a) = f + (1 − n·f)·μ_グラフ(a)                     （n は合法手の数）
 * ε = 節点の基本の割合 × 倍率 m
 * m = 最小 + (最大 − 最小)·p^k                                （p は迷いの百分位）
 * u = w·H(π)/log n + (1 − w)·2π₂/(π₁ + π₂)                     （迷い。π₁・π₂ は大きい方から 2 つの確率）
 * </pre>
 *
 * 基本の割合は、打つ牌を選ぶ節点が {@code discardMass}、ほかの節点（ロン・ツモ・九種九牌・鳴くか・鳴きの種類と候補・槓・槓の種類と
 * 候補・立直）が {@code gateMass}。百分位は、前の macro の迷いの分布（{@link UncertaintyCdf}）から求める。その分布がまだ無い間は、倍率を
 * 1 にする。今の macro の迷いは、次の macro のために数えておく（{@link #calibration}）。
 *
 * <p>合法手が 1 つの判断は探索しない（μ = 1）。どのスレッドから呼んでもよい。
 */
public final class Exploration {

  /** 方策の確率の下限。節点の条件つき確率を作れるように、合法手の確率を正にしておく。 */
  private static final float POLICY_FLOOR = 1e-8f;

  /** 合法手の確率の下限 f は、1 をこの値で割った値より小さくする。 */
  private static final int FLOOR_DIVISOR = 64;

  /**
   * 探索の設定。既定は Nano の自己対局の値。
   *
   * @param gateMass 打つ牌を選ぶ節点以外で、一様分布を混ぜる基本の割合
   * @param discardMass 打つ牌を選ぶ節点で、一様分布を混ぜる基本の割合
   * @param leafFloor 合法手の確率の下限 f（0 以上 1/64 未満）
   * @param minimumScale 倍率 m の最小値（0 より大きく 1 以下）
   * @param maximumScale 倍率 m の最大値（1 以上で、基本の割合に掛けた値は 1 未満）
   * @param exponent 百分位の指数 k（正の値）
   * @param entropyWeight 迷いに占めるエントロピーの重み w（0 以上 1 以下）
   */
  public record Settings(
      @Key(doc = "打つ牌を選ぶ節点以外で、一様分布を混ぜる基本の割合", value = "0.004") double gateMass,
      @Key(doc = "打つ牌を選ぶ節点で、一様分布を混ぜる基本の割合", value = "0.01") double discardMass,
      @Key(doc = "合法手の確率の下限（0 以上 1/64 未満）", value = "1e-4") double leafFloor,
      @Key(doc = "基本の割合に掛ける倍率の最小値（0 より大きく 1 以下）", value = "0.1") double minimumScale,
      @Key(doc = "基本の割合に掛ける倍率の最大値（1 以上で、基本の割合に掛けた値は 1 未満）", value = "5.0") double maximumScale,
      @Key(doc = "倍率を決める、迷いの百分位の指数（正の値）", value = "4.5") double exponent,
      @Key(doc = "迷いに占めるエントロピーの重み（0 以上 1 以下）", value = "0.5") double entropyWeight) {

    public Settings {
      if (!(gateMass >= 0) || !(discardMass >= 0) || !(Math.max(gateMass, discardMass) * maximumScale < 1)) {
        throw new IllegalArgumentException("Exploration masses times the maximum scale must be in [0, 1): " + this);
      }
      if (!(leafFloor >= 0 && leafFloor < 1.0 / FLOOR_DIVISOR)) {
        throw new IllegalArgumentException("The leaf floor must be in [0, 1/" + FLOOR_DIVISOR + "): " + leafFloor);
      }
      if (!(minimumScale > 0 && minimumScale <= 1) || !(maximumScale >= 1) || !(exponent > 0)
          || !Double.isFinite(exponent) || !(entropyWeight >= 0 && entropyWeight <= 1)) {
        throw new IllegalArgumentException("Bad exploration scale: " + this);
      }
    }
  }

  /** 合法手が 2 つ以上ある判断の種類。上の種類ほど優先する（ロンと鳴きを選べる応答は {@code AGARI_OR_ABORT}）。 */
  public enum Kind {
    /** ロン・ツモ・九種九牌を選べる。 */
    AGARI_OR_ABORT,
    /** チー・ポン・大明槓を選べる。 */
    CALL,
    /** 暗槓・加槓を選べる。 */
    KAN,
    /** 立直を選べる。 */
    RIICHI,
    /** 打牌だけ。 */
    DISCARD
  }

  private final Settings settings;
  private final UncertaintyCdf previous;
  private final UncertaintyCdf.Histogram current = new UncertaintyCdf.Histogram();
  private final ThreadLocal<Work> work = ThreadLocal.withInitial(Work::new);

  /**
   * @param previous 前の macro の迷いの分布（初めの macro は {@link UncertaintyCdf#empty}）
   */
  public Exploration(Settings settings, UncertaintyCdf previous) {
    this.settings = Objects.requireNonNull(settings, "settings");
    this.previous = Objects.requireNonNull(previous, "previous");
  }

  public Settings settings() {
    return settings;
  }

  /**
   * 判断で手を選ぶ振る舞いの確率を書き、方策の迷いを今の macro の分布に数える。
   *
   * @param observation 判断の記録
   * @param policy 方策の確率（合法手の順。和で割ってから使う）
   * @param tie 0 以上 1 未満の乱数（同じ迷いが並ぶ区間の中で百分位を散らす）
   * @param behaviour 振る舞いの確率を書く先（合法手の数以上の長さ）
   * @return 掛けた倍率 m
   * @throws IllegalArgumentException 方策の確率が負か有限でないか、和が正でない
   */
  public double behaviour(Observation observation, float[] policy, double tie, float[] behaviour) {
    int legal = observation.legalActions();
    if (legal == 1) {
      behaviour[0] = 1f;
      return 1;
    }
    Kind kind = kind(observation);
    boolean response = response(observation);
    double uncertainty = uncertainty(policy, legal, settings.entropyWeight());
    current.observe(kind, response, uncertainty);
    double percentile = previous.percentile(kind, response, uncertainty, tie);
    double scale =
        Double.isNaN(percentile)
            ? 1
            : settings.minimumScale()
                + (settings.maximumScale() - settings.minimumScale()) * Math.pow(percentile, settings.exponent());
    mix(observation, policy, scale, behaviour);
    return scale;
  }

  /**
   * 渡した倍率で一様分布を混ぜる。迷いは数えない。
   *
   * @param scale 倍率 m（0 以上で、最大の倍率以下）
   */
  public void mix(Observation observation, float[] policy, double scale, float[] behaviour) {
    if (!(scale >= 0 && scale <= settings.maximumScale())) {
      throw new IllegalArgumentException("The exploration scale must be in [0, " + settings.maximumScale() + "]: " + scale);
    }
    int legal = observation.legalActions();
    Work at = work.get();
    float[] canonical = at.canonical;
    double sum = 0;
    for (int action = 0; action < legal; action++) {
      float probability = policy[action];
      if (!(probability >= 0) || !Float.isFinite(probability)) {
        throw new IllegalArgumentException("A policy probability must be finite and non-negative: " + probability);
      }
      sum += probability;
    }
    if (!(sum > 0)) {
      throw new IllegalArgumentException("The policy probabilities must have a positive sum: " + sum);
    }
    double positive = 0;
    for (int action = 0; action < legal; action++) {
      canonical[action] = (float) Math.max(POLICY_FLOOR, policy[action] / sum);
      positive += canonical[action];
    }
    // 一様分布を混ぜない節点では、方策が 1 つの手に寄っていると、枝の確率が FP32 の丸めで 0 になりうる。0 は最後の下限で正にする。
    // グラフが確率を書かなかった手は負の値のまま残るので、それで見分ける
    for (int action = 0; action < legal; action++) {
      canonical[action] = (float) (canonical[action] / positive);
      behaviour[action] = -1f;
    }
    float gate = (float) (settings.gateMass() * scale);
    long all = (1L << legal) - 1;
    if (response(observation)) {
      respond(observation, canonical, all, gate, behaviour);
    } else {
      turn(observation, canonical, all, gate, (float) (settings.discardMass() * scale), at.identities, behaviour);
    }
    double total = 0;
    for (int action = 0; action < legal; action++) {
      if (!(behaviour[action] >= 0) || !Float.isFinite(behaviour[action])) {
        throw new IllegalStateException(
            "The policy graph left action " + observation.action(action) + " without probability");
      }
      total += behaviour[action];
    }
    double floor = settings.leafFloor();
    double retained = (1 - floor * legal) / total;
    for (int action = 0; action < legal; action++) {
      behaviour[action] = (float) (floor + retained * behaviour[action]);
    }
  }

  /** 今の macro で数えた迷いの分布（次の macro に渡す）。収集を終えた後に呼ぶ。 */
  public UncertaintyCdf calibration() {
    return current.freeze();
  }

  /**
   * 合法手が 2 つ以上ある判断の種類。
   *
   * @throws IllegalArgumentException 合法手が 1 つしかない判断のとき
   */
  public static Kind kind(Observation observation) {
    int legal = observation.legalActions();
    if (legal < 2) {
      throw new IllegalArgumentException("A forced decision has no exploration kind");
    }
    boolean call = false;
    boolean kan = false;
    boolean riichi = false;
    for (int action = 0; action < legal; action++) {
      switch (observation.action(action).type()) {
        case RON_AGARI, TSUMO_AGARI, KYUSHU_KYUHAI -> {
          return Kind.AGARI_OR_ABORT;
        }
        case CHI, PON, DAIMINKAN -> call = true;
        case ANKAN, KAKAN -> kan = true;
        case RIICHI_DAHAI -> riichi = true;
        case DAHAI, PASS, NUKI -> {}
      }
    }
    return call ? Kind.CALL : kan ? Kind.KAN : riichi ? Kind.RIICHI : Kind.DISCARD;
  }

  /**
   * 方策の迷い（0 以上 1 以下）。
   *
   * @param policy 合法手の確率（和で割ってから使う）
   * @param legal 合法手の数（2 以上）
   * @param entropyWeight エントロピーの重み w
   */
  public static double uncertainty(float[] policy, int legal, double entropyWeight) {
    double sum = 0;
    for (int action = 0; action < legal; action++) {
      sum += policy[action];
    }
    double entropy = 0;
    double first = -1;
    double second = -1;
    for (int action = 0; action < legal; action++) {
      double probability = policy[action] / sum;
      if (probability > 0) {
        entropy -= probability * Math.log(probability);
      }
      if (probability > first) {
        second = first;
        first = probability;
      } else if (probability > second) {
        second = probability;
      }
    }
    double normalized = clamp(entropy / Math.log(legal));
    double ambiguity = clamp(2 * second / (first + second + 1e-8));
    return clamp(entropyWeight * normalized + (1 - entropyWeight) * ambiguity);
  }

  private static boolean response(Observation observation) {
    EngineDecisionKind kind = observation.decisionKind();
    return kind == EngineDecisionKind.RESPONSE
        || kind == EngineDecisionKind.CHANKAN
        || kind == EngineDecisionKind.NUKI_RESPONSE;
  }

  /** 応答の節点を、ロンするか → 鳴くか → 鳴きの種類 → 種類の中の候補、の順にたどる。 */
  private static void respond(Observation o, float[] canonical, long all, float gate, float[] out) {
    long ron = slots(o, all, Action.Type.RON_AGARI);
    long declined = all & ~ron;
    long pass = slots(o, declined, Action.Type.PASS);
    long meld = declined & ~pass;
    long chi = slots(o, meld, Action.Type.CHI);
    long pon = slots(o, meld, Action.Type.PON);
    long daiminkan = slots(o, meld, Action.Type.DAIMINKAN);
    float ronAccepted = binary(canonical, ron, declined, gate);
    assign(canonical, ron, ronAccepted, 0f, out);
    float callAccepted = binary(canonical, meld, pass, gate);
    assign(canonical, pass, (1f - ronAccepted) * (1f - callAccepted), 0f, out);
    float typeMass = mass(canonical, meld);
    int types = present(chi) + present(pon) + present(daiminkan);
    float branch = (1f - ronAccepted) * callAccepted;
    assign(canonical, chi, branch * choice(canonical, chi, typeMass, types, gate), gate, out);
    assign(canonical, pon, branch * choice(canonical, pon, typeMass, types, gate), gate, out);
    assign(canonical, daiminkan, branch * choice(canonical, daiminkan, typeMass, types, gate), gate, out);
  }

  /** 手番の節点を、ツモ和了するか → 九種九牌か → 槓か打牌か、の順にたどる。槓は種類 → 候補、打牌は打つ牌 → 立直か黙聴か。 */
  private static void turn(
      Observation o, float[] canonical, long all, float gate, float discard, int[] identities, float[] out) {
    long tsumo = slots(o, all, Action.Type.TSUMO_AGARI);
    long afterTsumo = all & ~tsumo;
    float tsumoAccepted = binary(canonical, tsumo, afterTsumo, gate);
    assign(canonical, tsumo, tsumoAccepted, 0f, out);
    long kyushu = slots(o, afterTsumo, Action.Type.KYUSHU_KYUHAI);
    long afterKyushu = afterTsumo & ~kyushu;
    float kyushuAccepted = binary(canonical, kyushu, afterKyushu, gate);
    assign(canonical, kyushu, (1f - tsumoAccepted) * kyushuAccepted, 0f, out);
    long discards = slots(o, afterKyushu, Action.Type.DAHAI) | slots(o, afterKyushu, Action.Type.RIICHI_DAHAI);
    long ankan = slots(o, afterKyushu, Action.Type.ANKAN);
    long kakan = slots(o, afterKyushu, Action.Type.KAKAN);
    long kan = ankan | kakan;
    float kanAccepted = binary(canonical, kan, discards, gate);
    float path = (1f - tsumoAccepted) * (1f - kyushuAccepted);
    float kanMass = mass(canonical, kan);
    int kanTypes = present(ankan) + present(kakan);
    assign(canonical, ankan, path * kanAccepted * choice(canonical, ankan, kanMass, kanTypes, gate), gate, out);
    assign(canonical, kakan, path * kanAccepted * choice(canonical, kakan, kanMass, kanTypes, gate), gate, out);
    discards(o, canonical, discards, path * (1f - kanAccepted), gate, discard, identities, out);
  }

  /**
   * 打つ牌を選ぶ節点と、立直か黙聴かの節点。同じ牌を同じ打ち方（赤か、ツモ切りか）で打つ打牌と立直の打牌は、打つ牌として 1 つの候補に
   * まとめる。
   */
  private static void discards(
      Observation o,
      float[] canonical,
      long slots,
      float branch,
      float gate,
      float discard,
      int[] identities,
      float[] out) {
    int count = 0;
    for (long remaining = slots; remaining != 0; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      identities[slot] = VocabularyCodec.discardIdentity(o.action(slot));
    }
    for (long remaining = slots; remaining != 0; count++) {
      remaining &= ~same(identities, remaining);
    }
    float identityMass = mass(canonical, slots);
    for (long remaining = slots; remaining != 0; ) {
      long same = same(identities, remaining);
      long riichi = slots(o, same, Action.Type.RIICHI_DAHAI);
      long dama = same & ~riichi;
      float identity = choice(canonical, same, identityMass, count, discard);
      float riichiAccepted = binary(canonical, riichi, dama, gate);
      assign(canonical, riichi, branch * identity * riichiAccepted, 0f, out);
      assign(canonical, dama, branch * identity * (1f - riichiAccepted), 0f, out);
      remaining &= ~same;
    }
  }

  /** 渡した位置のうち、いちばん前の手と同じ牌を同じ打ち方で打つ手の位置（bit の集合）。 */
  private static long same(int[] identities, long slots) {
    int identity = identities[Long.numberOfTrailingZeros(slots)];
    long same = 0;
    for (long remaining = slots; remaining != 0; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      if (identities[slot] == identity) {
        same |= 1L << slot;
      }
    }
    return same;
  }

  /** 二択で選ぶ側の確率。選ぶ側が無ければ 0、選ばない側が無ければ 1。 */
  private static float binary(float[] canonical, long accepted, long declined, float exploration) {
    if (accepted == 0) {
      return 0f;
    }
    if (declined == 0) {
      return 1f;
    }
    float acceptedMass = mass(canonical, accepted);
    return mixed(acceptedMass / (acceptedMass + mass(canonical, declined)), 0.5f, exploration);
  }

  /** 候補の中の 1 つを選ぶ確率（候補が無ければ 0）。 */
  private static float choice(float[] canonical, long slots, float total, int alternatives, float exploration) {
    if (slots == 0) {
      return 0f;
    }
    return mixed(mass(canonical, slots) / total, 1f / alternatives, exploration);
  }

  /** 枝の確率を枝の中の手へ割り当てる。手ごとの割合は、方策の条件つき確率に一様分布を混ぜた値。 */
  private static void assign(float[] canonical, long slots, float branch, float exploration, float[] out) {
    if (slots == 0) {
      return;
    }
    float slotsMass = mass(canonical, slots);
    float probe = 1f / Long.bitCount(slots);
    for (long remaining = slots; remaining != 0; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      out[slot] = branch * mixed(canonical[slot] / slotsMass, probe, exploration);
    }
  }

  private static float mixed(float policy, float probe, float exploration) {
    return policy * (1f - exploration) + probe * exploration;
  }

  private static float mass(float[] canonical, long slots) {
    float mass = 0f;
    for (long remaining = slots; remaining != 0; remaining &= remaining - 1) {
      mass += canonical[Long.numberOfTrailingZeros(remaining)];
    }
    return mass;
  }

  private static long slots(Observation o, long eligible, Action.Type type) {
    long found = 0;
    for (long remaining = eligible; remaining != 0; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      if (o.action(slot).type() == type) {
        found |= 1L << slot;
      }
    }
    return found;
  }

  private static int present(long slots) {
    return slots == 0 ? 0 : 1;
  }

  private static double clamp(double value) {
    return Math.max(0, Math.min(1, value));
  }

  /** スレッドごとの作業領域。正規化した方策の確率と、物理の打牌の番号を置く。 */
  private static final class Work {
    private final float[] canonical = new float[ClassicSchema.MAX_LEGAL];
    private final int[] identities = new int[ClassicSchema.MAX_LEGAL];
  }
}
