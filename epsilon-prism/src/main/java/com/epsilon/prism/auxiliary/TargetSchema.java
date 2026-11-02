package com.epsilon.prism.auxiliary;

import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.input.InputSchema.ActionRole;

/**
 * PRISM の補助の頭の教師を宣言する。補助の頭は学習のときだけ計算し、推論の費用は 0 にする。
 *
 * <p>候補の頭（{@link CandidateTarget}）は、測った手の経路の局末の結果を、判断の席から見た教師にする。どの（行, 手）にも「o で a を打ち、
 * 以後は全席が b」の経路がちょうど 1 本あるので、教師は E[· | o, a] の偏りのない 1 標本になる。読みの頭（{@link ReadingTarget}）は、判断の
 * 時点の全員の手から作る神託を、相手 3 席ごとの教師にする。
 *
 * <p>頭ごとの重みは持たず、損失の重みは群（候補・読み）ごとの α だけにする。頭の出力は enum の宣言の順に区画を並べ、区画の先頭と幅の和は
 * enum から一度だけ作る。
 *
 * <p>役の教師は束ねない。{@link CandidateTarget#YAKU} の列 k は、エンジンの役 {@link ScoringYaku} の ordinal k の役（役のビットの k 番目、
 * {@link ScoringYaku#bit()}）で、自風・場風・三元牌の役牌も別々の列になる。
 *
 * <p>{@link #fingerprint()} は頭の名前・順・幅・損失・列のまとめ方・条件・正規化と、役の名前と順・候補の教師を使う手の役割から作る。
 * 入力の指紋とは分けるので、頭を変えても推論の入力は合ったままになる。
 */
public final class TargetSchema {

  /** 相手の向聴の類の数（0・1・2・3 以上）。 */
  public static final int SHANTEN_CLASSES = 4;

  /** 点の教師を割る数。 */
  private static final float POINT_SCALE = 8000f;

  private TargetSchema() {}

  /**
   * 補助の頭の指紋を返す。model.json に書き、開くときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /**
   * 相手の最小向聴を類に写す。
   *
   * @param shanten 最小向聴（0 以上）
   * @return 類（3 以上は 3）
   * @throws IllegalArgumentException 向聴が負のとき
   */
  public static int shantenClass(int shanten) {
    if (shanten < 0) {
      throw new IllegalArgumentException("negative shanten: " + shanten);
    }
    return Math.min(shanten, SHANTEN_CLASSES - 1);
  }

  /**
   * 手の役割が候補の教師（{@link CandidateTarget}）を使うなら true を返す。和了と九種九牌は局の終わり方が手で決まり、結果が自明なので
   * 使わない。記録を省く配線と教師を作る側は、この写し方だけを引く。default を書かないので、役割を足すと組み立てで落ちる。
   *
   * @param role 手の役割
   * @return 候補の教師を使うなら true
   */
  public static boolean hasCandidateTargets(ActionRole role) {
    return switch (role) {
      case DISCARD, RIICHI, CALL, KAN, PASS -> true;
      case TSUMO, RON, KYUSHU -> false;
    };
  }

  /** 測った手の経路の局の終わり方（判断の席から見る）。和了率は自分のツモ＋自分のロン、放銃率は放銃。 */
  public enum Outcome {
    SELF_TSUMO,
    /** 自分のロン和了（複数ロンの 1 人を含む）。 */
    SELF_RON,
    /** 自分の放銃（槍槓を含む）。 */
    DEAL_IN,
    OTHER_TSUMO,
    /** 他家どうしのロン。 */
    OTHER_RON,
    /** 流局で自分は聴牌。 */
    DRAW_TENPAI,
    /** 流局で自分は不聴。 */
    DRAW_NOTEN,
    ABORTIVE
  }

  /** 損失の形。 */
  public enum Loss {
    /** 区画の上の softmax の交差エントロピー。 */
    SOFTMAX,
    /** 列ごとの sigmoid の 2 値の交差エントロピー。 */
    SIGMOID,
    /** 列ごとの二乗誤差（教師は scale で割る）。 */
    SQUARED
  }

  /** 区画の中の列ごとの損失のまとめ方（SIGMOID・SQUARED の多い列の区画で意味を持つ。SOFTMAX は区画で 1 つ）。 */
  public enum Reduction {
    /** 列の平均。牌ごとのように、どの列もよく 1 になる区画に使う。 */
    MEAN,
    /** 列の和。役のように、ほとんどの列がまれにしか 1 にならない区画に使う（まれな列の損失はほぼ 0 なので、よく出る列が薄まらない）。 */
    SUM
  }

  /** 候補の教師を使う条件（判断の席から見た {@link Outcome}）。満たさない（行, 手）は損失にも分母にも入れない。 */
  public enum Condition {
    ALWAYS,
    /** 自分の和了（ツモ・ロン）。 */
    SELF_WIN,
    /** 自分の放銃。 */
    DEAL_IN,
    /** 他家の和了（放銃を含む）。 */
    OTHER_WIN;

    /**
     * 局の終わり方がこの条件を満たすなら true を返す。
     *
     * @param outcome 局の終わり方
     * @return 満たすなら true
     */
    public boolean holds(Outcome outcome) {
      return switch (this) {
        case ALWAYS -> true;
        case SELF_WIN -> outcome == Outcome.SELF_TSUMO || outcome == Outcome.SELF_RON;
        case DEAL_IN -> outcome == Outcome.DEAL_IN;
        case OTHER_WIN ->
            outcome == Outcome.DEAL_IN
                || outcome == Outcome.OTHER_TSUMO
                || outcome == Outcome.OTHER_RON;
      };
    }
  }

  /** 読みの教師を使う条件（相手ごと、牌ごと）。満たさない所は損失にも分母にも入れない。 */
  public enum ReadingCondition {
    ALWAYS,
    /** 相手が聴牌のときだけ。 */
    OPPONENT_TENPAI,
    /** その牌で相手がロンできるときだけ。 */
    RON_TILE
  }

  /** 頭の出力の 1 区画。 */
  public sealed interface Target permits CandidateTarget, ReadingTarget {

    /**
     * 損失の形を返す。
     *
     * @return 損失の形
     */
    Loss loss();

    /**
     * 区画の中の列ごとの損失のまとめ方を返す。
     *
     * @return まとめ方
     */
    Reduction reduction();

    /**
     * 区画の幅を返す。
     *
     * @return 列の数
     */
    int width();

    /**
     * 区画の先頭（頭の出力の中の列）を返す。
     *
     * @return 列
     */
    int offset();

    /**
     * 教師を割る数を返す（点の教師は 8000、ほかは 1）。
     *
     * @return 正規化の定数
     */
    float scale();

    /**
     * 教師を使う条件を返す。
     *
     * @return 条件
     */
    Enum<?> condition();
  }

  /** 候補ごとの教師（測った手の経路の局末から）。使う手の役割は {@link #hasCandidateTargets} が決める。 */
  public enum CandidateTarget implements Target {
    /** 局の終わり方（{@link Outcome} の 8 類）。 */
    OUTCOME(Loss.SOFTMAX, Reduction.MEAN, Outcome.values().length, Condition.ALWAYS, 1f),
    /** 判断の後、局の終わりまでに聴牌に届いた（和了も 1）。 */
    TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, Condition.ALWAYS, 1f),
    /** 自分の和了の役。列 k は {@link ScoringYaku} の ordinal k の役で、束ねない。列の損失は和にする。 */
    YAKU(Loss.SIGMOID, Reduction.SUM, ScoringYaku.values().length, Condition.SELF_WIN, 1f),
    /** 自分の和了で受け取る点（本場・供託を除く）。 */
    WIN_POINTS(Loss.SQUARED, Reduction.MEAN, 1, Condition.SELF_WIN, POINT_SCALE),
    /** 放銃で払う点の和（本場を除く）。 */
    DEAL_IN_POINTS(Loss.SQUARED, Reduction.MEAN, 1, Condition.DEAL_IN, POINT_SCALE),
    /** 局の終わりまでにロンの役がある聴牌（立直は役に数える）に届いた（和了も 1）。 */
    YAKU_TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, Condition.ALWAYS, 1f),
    /** 局末の点 − 判断の時点の点。 */
    POINT_DELTA(Loss.SQUARED, Reduction.MEAN, 1, Condition.ALWAYS, POINT_SCALE),
    /**
     * 和了した他家の相対席（下家・対面・上家）。他家 2 人のロン（ダブロン）では、供託を受け取る和了者と同じく、放銃者の下家から順に数えて
     * 最初の和了者（頭跳ねの順）を教師にする。
     */
    WINNER(Loss.SOFTMAX, Reduction.MEAN, GameState.NUM_PLAYERS - 1, Condition.OTHER_WIN, 1f);

    /** 候補の頭の出力の幅（全ての区画の幅の和）。 */
    public static final int WIDTH;

    private static final int[] OFFSETS;

    static {
      CandidateTarget[] targets = values();
      OFFSETS = new int[targets.length];
      int offset = 0;
      for (CandidateTarget target : targets) {
        OFFSETS[target.ordinal()] = offset;
        offset += target.width;
      }
      WIDTH = offset;
    }

    private final Loss loss;
    private final Reduction reduction;
    private final int width;
    private final Condition condition;
    private final float scale;

    CandidateTarget(Loss loss, Reduction reduction, int width, Condition condition, float scale) {
      this.loss = loss;
      this.reduction = reduction;
      this.width = width;
      this.condition = condition;
      this.scale = scale;
    }

    @Override
    public Loss loss() {
      return loss;
    }

    @Override
    public Reduction reduction() {
      return reduction;
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public int offset() {
      return OFFSETS[ordinal()];
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Condition condition() {
      return condition;
    }
  }

  /** 行ごと・相手 3 席ごとの教師（判断の時点の全員の手から作る神託）。牌の軸には入力と同じ色の写しをかける。 */
  public enum ReadingTarget implements Target {
    /** 相手が聴牌（形の聴牌）。 */
    TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, ReadingCondition.ALWAYS, 1f),
    /** 相手の最小向聴の類（{@link TargetSchema#shantenClass}）。 */
    SHANTEN(Loss.SOFTMAX, Reduction.MEAN, SHANTEN_CLASSES, ReadingCondition.ALWAYS, 1f),
    /** 今その牌を捨てたら相手がロンできる（役・振聴・5 枚目まで見る）。 */
    DANGER(Loss.SIGMOID, Reduction.MEAN, Tile.NUM_TILE_TYPES, ReadingCondition.ALWAYS, 1f),
    /** 相手の形の待ち。 */
    WAITS(Loss.SIGMOID, Reduction.MEAN, Tile.NUM_TILE_TYPES, ReadingCondition.OPPONENT_TENPAI, 1f),
    /** その牌のロンの和了の点（裏なし。包の分け方は含まない。{@link OracleCodec} の決まり）。 */
    DANGER_POINTS(
        Loss.SQUARED, Reduction.MEAN, Tile.NUM_TILE_TYPES, ReadingCondition.RON_TILE, POINT_SCALE);

    /** 相手 1 席あたりの読みの頭の出力の幅（全ての区画の幅の和）。 */
    public static final int WIDTH;

    private static final int[] OFFSETS;

    static {
      ReadingTarget[] targets = values();
      OFFSETS = new int[targets.length];
      int offset = 0;
      for (ReadingTarget target : targets) {
        OFFSETS[target.ordinal()] = offset;
        offset += target.width;
      }
      WIDTH = offset;
    }

    private final Loss loss;
    private final Reduction reduction;
    private final int width;
    private final ReadingCondition condition;
    private final float scale;

    ReadingTarget(
        Loss loss, Reduction reduction, int width, ReadingCondition condition, float scale) {
      this.loss = loss;
      this.reduction = reduction;
      this.width = width;
      this.condition = condition;
      this.scale = scale;
    }

    @Override
    public Loss loss() {
      return loss;
    }

    @Override
    public Reduction reduction() {
      return reduction;
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public int offset() {
      return OFFSETS[ordinal()];
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public ReadingCondition condition() {
      return condition;
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE = digest().hex();

    private static SchemaDigest digest() {
      StringBuilder roles = new StringBuilder();
      for (ActionRole role : ActionRole.values()) {
        roles.append(role).append('=').append(hasCandidateTargets(role)).append(',');
      }
      SchemaDigest digest =
          SchemaDigest.named("epsilon-prism-v2-heads")
              .values(Outcome.class)
              .values(ScoringYaku.class)
              .values(Loss.class)
              .values(Reduction.class)
              .values(Condition.class)
              .values(ReadingCondition.class)
              .with("candidateRoles", roles);
      for (Class<?> type : SchemaDigest.sorted(Target.class.getPermittedSubclasses())) {
        StringBuilder targets = new StringBuilder();
        for (Object constant : type.getEnumConstants()) {
          Target target = (Target) constant;
          targets.append(constant).append(':').append(target.loss()).append(':');
          targets.append(target.reduction()).append(':').append(target.width()).append(':');
          targets.append(target.condition()).append('/').append(target.scale()).append(',');
        }
        digest.with(SchemaDigest.name(type), targets);
      }
      return digest.with("shantenClasses", SHANTEN_CLASSES);
    }
  }
}
