package com.epsilon.prism.auxiliary;

import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.input.TileTokens;

/**
 * PRISM の補助のヘッドが学ぶ教師を宣言する。補助のヘッドは学習のときだけ計算するので、推論のコストは増えない。
 *
 * <p>候補のヘッド（{@link CandidateTarget}）は、測った手を打った後の経路の局末を、判断した席から見た教師にする。どの（行, 手）にも
 * 「o で a を打ち、以後は全席が b に従う」経路がちょうど 1 本あるので、教師は E[· | o, a] の偏りのない 1 標本になる。読みのヘッド
 * （{@link ReadingTarget}）は、判断の時点の全員の手から作る Oracle の値を、相手ごとの教師にする。
 *
 * <p>ヘッドごとの重みは持たない。損失の重みは、群（候補・読み）ごとの α だけにする。ヘッドの出力は、教師ごとの列の範囲を enum の宣言の
 * 順に並べたもの。範囲の幅はルールで決まる（和了した他家は席の数 − 1、読みの牌ごとの教師は牌のトークンの数 {@link TileTokens#count}）。
 * 範囲の先頭と幅の合計は、ルールごとに 1 回だけ求めておき、{@link #of} が返す形が持つ。
 *
 * <p>役の教師はまとめない。{@link CandidateTarget#YAKU} の列 k は {@link ScoringYaku} の ordinal k の役（役のビットの k 番目、
 * {@link ScoringYaku#bit()}）で、自風・場風・三元牌の役牌も別々の列になる。
 *
 * <p>{@link #fingerprint()} は、教師の名前・順・幅・損失・列のまとめ方・条件・割る数、役の名前と順、候補の教師を使う手の種類から作る。
 * 入力のフィンガープリントとは分けるので、ヘッドを変えても推論の入力は合ったままになる。
 */
public final class TargetSchema {

  /** 相手の向聴の区分の数（0・1・2・3 以上）。 */
  public static final int SHANTEN_CLASSES = 4;

  /** 点の教師を割る数。 */
  private static final float POINT_SCALE = 8000f;

  /** 読みの教師が牌ごとであることを示す幅の値。実際の幅は、そのルールの牌のトークンの数になる。 */
  private static final int PER_TILE = 0;

  private static final CandidateTarget[] CANDIDATES = CandidateTarget.values();
  private static final ReadingTarget[] READINGS = ReadingTarget.values();

  private static final TargetSchema[] SCHEMAS = new TargetSchema[Variant.values().length];

  static {
    for (Variant variant : Variant.values()) {
      SCHEMAS[variant.ordinal()] = new TargetSchema(variant);
    }
  }

  private final Variant variant;
  private final int[] candidateOffsets = new int[CANDIDATES.length];
  private final int[] candidateWidths = new int[CANDIDATES.length];
  private final int[] readingOffsets = new int[READINGS.length];
  private final int[] readingWidths = new int[READINGS.length];
  private final int candidateWidth;
  private final int readingWidth;

  /** 入力の形式が要るので、初めて使うときに作る。 */
  private volatile String fingerprint;

  private TargetSchema(Variant variant) {
    this.variant = variant;
    int offset = 0;
    for (CandidateTarget target : CANDIDATES) {
      candidateOffsets[target.ordinal()] = offset;
      candidateWidths[target.ordinal()] = target.width(variant);
      offset += candidateWidths[target.ordinal()];
    }
    candidateWidth = offset;
    offset = 0;
    for (ReadingTarget target : READINGS) {
      readingOffsets[target.ordinal()] = offset;
      readingWidths[target.ordinal()] = target.width(variant);
      offset += readingWidths[target.ordinal()];
    }
    readingWidth = offset;
  }

  /**
   * ルールごとの補助のヘッドの形を返す。
   *
   * @param variant ルール
   * @return そのルールの形
   */
  public static TargetSchema of(Variant variant) {
    return SCHEMAS[variant.ordinal()];
  }

  public Variant variant() {
    return variant;
  }

  /** 読みのヘッドが見る相手の数（席の数 − 1）。 */
  public int opponents() {
    return variant.seats() - 1;
  }

  /** 候補のヘッドの出力の列の数（全ての教師の幅の合計）。 */
  public int candidateWidth() {
    return candidateWidth;
  }

  /** 読みのヘッドの、相手 1 人あたりの出力の列の数（全ての教師の幅の合計）。 */
  public int readingWidth() {
    return readingWidth;
  }

  /**
   * 教師の列の範囲の幅を返す。
   *
   * @param target 教師
   * @return 列の数
   */
  public int width(Target target) {
    return switch (target) {
      case CandidateTarget candidate -> candidateWidths[candidate.ordinal()];
      case ReadingTarget reading -> readingWidths[reading.ordinal()];
    };
  }

  /**
   * 教師の列の範囲の先頭を返す。
   *
   * @param target 教師
   * @return ヘッドの出力の中の列の番号
   */
  public int offset(Target target) {
    return switch (target) {
      case CandidateTarget candidate -> candidateOffsets[candidate.ordinal()];
      case ReadingTarget reading -> readingOffsets[reading.ordinal()];
    };
  }

  /**
   * 補助のヘッドのフィンガープリントを返す。model.json に書き、開くときに照合する。
   *
   * @return SHA-256 の先頭 8 B を 16 進で書いた文字列
   * @throws IllegalStateException このルールの入力の形式がないとき
   */
  public String fingerprint() {
    String value = fingerprint;
    if (value == null) {
      value = digest().hex();
      fingerprint = value;
    }
    return value;
  }

  /**
   * 相手の最小向聴を区分に直す。
   *
   * @param shanten 最小向聴（0 以上）
   * @return 区分（3 以上は 3）
   * @throws IllegalArgumentException 向聴が負のとき
   */
  public static int shantenClass(int shanten) {
    if (shanten < 0) {
      throw new IllegalArgumentException("negative shanten: " + shanten);
    }
    return Math.min(shanten, SHANTEN_CLASSES - 1);
  }

  /**
   * その種類の手が候補の教師（{@link CandidateTarget}）を使うなら true を返す。和了と九種九牌は、選んだ時点で局の終わり方が決まり、
   * 結果が自明なので使わない。記録を省く側も教師を作る側も、このメソッドだけで決める。
   *
   * @param role 手の種類
   * @return 候補の教師を使うなら true
   */
  public static boolean hasCandidateTargets(Role role) {
    return !role.endsHand();
  }

  private SchemaDigest digest() {
    StringBuilder roles = new StringBuilder();
    for (Role role : InputFormat.of(variant).roles()) {
      roles.append(role).append('=').append(hasCandidateTargets(role)).append(',');
    }
    SchemaDigest digest =
        SchemaDigest.named("epsilon-prism-v2-heads")
            .values(Vocabulary.of(Outcome.class, variant))
            .values(Vocabulary.of(ScoringYaku.class, variant))
            .values(Vocabulary.of(Loss.class, variant))
            .values(Vocabulary.of(Reduction.class, variant))
            .values(Vocabulary.of(Condition.class, variant))
            .values(Vocabulary.of(ReadingCondition.class, variant))
            .with("candidateRoles", roles);
    for (Class<?> type : SchemaDigest.sorted(Target.class.getPermittedSubclasses())) {
      StringBuilder targets = new StringBuilder();
      for (Object constant : type.getEnumConstants()) {
        Target target = (Target) constant;
        targets.append(constant).append(':').append(target.loss()).append(':');
        targets.append(target.reduction()).append(':').append(width(target)).append(':');
        targets.append(target.condition()).append('/').append(target.scale()).append(',');
      }
      digest.with(SchemaDigest.name(type), targets);
    }
    return digest.with("shantenClasses", SHANTEN_CLASSES);
  }

  /** 測った手を打った後の経路で、局がどう終わったか（判断した席から見る）。和了率は自分のツモ和了とロン和了の率の和、放銃率は放銃の率。 */
  public enum Outcome {
    SELF_TSUMO,
    /** 自分のロン和了（複数ロンの 1 人を含む）。 */
    SELF_RON,
    /** 自分の放銃（槍槓を含む）。 */
    DEAL_IN,
    OTHER_TSUMO,
    /** 他家が他家からロン和了した。 */
    OTHER_RON,
    /** 流局で自分は聴牌。 */
    DRAW_TENPAI,
    /** 流局で自分は不聴。 */
    DRAW_NOTEN,
    ABORTIVE
  }

  /** 損失の種類。 */
  public enum Loss {
    /** 範囲の列にかけた softmax の交差エントロピー。 */
    SOFTMAX,
    /** 列ごとの sigmoid の 2 値の交差エントロピー。 */
    SIGMOID,
    /** 列ごとの二乗誤差（教師は scale で割る）。 */
    SQUARED
  }

  /** 範囲の中の、列ごとの損失のまとめ方。意味を持つのは、列が複数ある sigmoid・二乗誤差の範囲だけ（softmax は範囲ごとに損失が 1 つ）。 */
  public enum Reduction {
    /** 列の平均。牌ごとの教師のように、どの列もよく 1 になる範囲に使う。 */
    MEAN,
    /**
     * 列の和。役のように、ほとんどの列がまれにしか 1 にならない範囲に使う。まれな列の損失はほぼ 0 なので、和にすればよく出る列が
     * 薄まらない。
     */
    SUM
  }

  /** 候補の教師を使う条件（判断した席から見た {@link Outcome}）。満たさない（行, 手）は損失にも分母にも入れない。 */
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

  /** 読みの教師を使う条件（相手ごと、牌ごと）。満たさない相手・牌は損失にも分母にも入れない。 */
  public enum ReadingCondition {
    ALWAYS,
    /** 相手が聴牌のときだけ。 */
    OPPONENT_TENPAI,
    /** その牌で相手がロンできるときだけ。 */
    RON_TILE
  }

  /** 1 つの教師。ヘッドの出力の列の範囲を 1 つ使い、その幅と先頭は {@link TargetSchema#width}・{@link TargetSchema#offset} が返す。 */
  public sealed interface Target permits CandidateTarget, ReadingTarget {

    /** 損失の種類。 */
    Loss loss();

    /** 範囲の中の、列ごとの損失のまとめ方。 */
    Reduction reduction();

    /**
     * そのルールでの範囲の幅を返す。
     *
     * @param variant ルール
     * @return 列の数
     */
    int width(Variant variant);

    /** 教師を割る数（点の教師は 8000、ほかは 1）。 */
    float scale();

    /** 教師を使う条件。 */
    Enum<?> condition();
  }

  /** 候補ごとの教師で、測った手を打った後の経路の局末から作る。どの種類の手で使うかは {@link #hasCandidateTargets} が決める。 */
  public enum CandidateTarget implements Target {
    /** 局の終わり方（{@link Outcome} の 8 通り）。 */
    OUTCOME(Loss.SOFTMAX, Reduction.MEAN, Outcome.values().length, Condition.ALWAYS, 1f),
    /** 判断の後、局の終わりまでに聴牌に届いた（和了も 1）。 */
    TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, Condition.ALWAYS, 1f),
    /** 自分の和了の役。役はまとめず、列 k が {@link ScoringYaku} の ordinal k の役。列の損失は和にする。 */
    YAKU(Loss.SIGMOID, Reduction.SUM, ScoringYaku.values().length, Condition.SELF_WIN, 1f),
    /** 自分の和了で受け取る点（本場・供託を除く）。 */
    WIN_POINTS(Loss.SQUARED, Reduction.MEAN, 1, Condition.SELF_WIN, POINT_SCALE),
    /** 放銃で払う点の和（本場を除く）。 */
    DEAL_IN_POINTS(Loss.SQUARED, Reduction.MEAN, 1, Condition.DEAL_IN, POINT_SCALE),
    /** 判断の後、局の終わりまでにロンの役がある聴牌に届いた（立直は役に数える。和了も 1）。 */
    YAKU_TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, Condition.ALWAYS, 1f),
    /** 局末の点 − 判断の時点の点。 */
    POINT_DELTA(Loss.SQUARED, Reduction.MEAN, 1, Condition.ALWAYS, POINT_SCALE),
    /**
     * 和了した他家の相対席（下家から順に、席の数 − 1 列）。他家 2 人がロンした（ダブロン）ときは、放銃者の下家から順に数えて最初の
     * 和了者（頭ハネの順。供託を受け取る和了者と同じ）を教師にする。
     */
    WINNER(Loss.SOFTMAX, Reduction.MEAN, 0, Condition.OTHER_WIN, 1f);

    private final Loss loss;
    private final Reduction reduction;

    /** ルールに依らない幅。和了した他家の教師だけは、幅を席の数から決める。 */
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
    public int width(Variant variant) {
      return this == WINNER ? variant.seats() - 1 : width;
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

  /**
   * 行ごと・相手ごとの教師で、判断の時点の全員の手から作る Oracle の値。牌ごとの教師には入力と同じ色の入れ替えをかけ、牌の
   * トークンの順に並べる。
   */
  public enum ReadingTarget implements Target {
    /** 相手が聴牌（形の聴牌）。 */
    TENPAI(Loss.SIGMOID, Reduction.MEAN, 1, ReadingCondition.ALWAYS, 1f),
    /** 相手の最小向聴の区分（{@link TargetSchema#shantenClass}）。 */
    SHANTEN(Loss.SOFTMAX, Reduction.MEAN, SHANTEN_CLASSES, ReadingCondition.ALWAYS, 1f),
    /** 今その牌を捨てたら相手がロンできる。役・振聴・5 枚目が無いことまで見る。 */
    DANGER(Loss.SIGMOID, Reduction.MEAN, PER_TILE, ReadingCondition.ALWAYS, 1f),
    /** 相手の形の待ち。 */
    WAITS(Loss.SIGMOID, Reduction.MEAN, PER_TILE, ReadingCondition.OPPONENT_TENPAI, 1f),
    /** その牌でロンしたときの和了の点。裏ドラは数えず、包による分け方も含めない（{@link OracleCodec} と同じ決まり）。 */
    DANGER_POINTS(Loss.SQUARED, Reduction.MEAN, PER_TILE, ReadingCondition.RON_TILE, POINT_SCALE);

    private final Loss loss;
    private final Reduction reduction;

    /** ルールに依らない幅。牌ごとの教師は {@link #PER_TILE}。 */
    private final int width;
    private final ReadingCondition condition;
    private final float scale;

    ReadingTarget(Loss loss, Reduction reduction, int width, ReadingCondition condition, float scale) {
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
    public int width(Variant variant) {
      return width == PER_TILE ? TileTokens.of(variant).count() : width;
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
}
