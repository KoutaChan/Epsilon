package com.epsilon.prism.model;

import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.hip.CandidateStages;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipCapabilities;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.hip.HipLayer;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.TileTokens;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 局・席・牌とイベントをトークンとして処理するネットワークの形。
 * 形の型が保存形式を決める。model.json に保存し、開くときに重みと照合する。
 *
 * <p>Attention の幅 A = H/2（1 ヘッド 16 次元で H/32 ヘッド）と、RMSNorm の ε = 1e-5·256/H は、幅 H から決まる。
 *
 * <p>バックボーンの各層の役目と FFN の幅は {@link TokenTrunk#layers()} が決める。{@link V2} では、最後の層だけが局・席・牌のトークンをクエリにし、
 * ほかの層は全てのトークンをクエリにする。FFN は全ての層と候補層で同じ幅 F を使う。{@link V3} では、層 0 だけが全てのトークンをクエリに
 * してイベントのトークンも書き換える（FFN の幅は F）。層 1 から後は局・席・牌のトークンだけをクエリにし、イベントのトークンは層 0 の出力の
 * ままキーとバリューとして読む。
 */
public sealed interface PrismShape permits PrismShape.TokenTrunk {

  /** この形の入力を符号化する形式。 */
  default InputFormat inputFormat(InputFormat base) {
    return InputFormat.of(base.variant(), inputFeatures());
  }

  /** この形が使う入力の機能。 */
  default Set<InputFeature> inputFeatures() {
    return Set.of();
  }

  /** 候補層で基本の注意に加える計算を作る。 */
  default CandidateLayer.Features candidateFeatures(InputFormat format) {
    return CandidateLayer.Features.STANDARD;
  }

  /** 各層の注意と FFN を組み立てる。 */
  default EncoderLayer encoder(int layer, InputFormat format) {
    return new EncoderLayer(this, layer, format);
  }

  int HEAD_WIDTH = 16;

  /** RMSNorm の ε の、幅 {@link #NORM_REFERENCE_WIDTH} での値。 */
  double NORM_EPSILON = 1e-5;

  /**
   * RMSNorm の ε の基準の幅。ε は幅に反比例させ、この幅で {@link #NORM_EPSILON} になる。配布済みの V2・V3 の重みはこの ε で学んだので、
   * 値は変えない。
   */
  int NORM_REFERENCE_WIDTH = 256;

  /** バックボーンの層の数の下限。 */
  int MIN_DEPTH = 2;

  /** バックボーンの層の数の上限。 */
  int MAX_DEPTH = 3;

  /** バックボーンの層の数の既定値。 */
  int DEFAULT_DEPTH = 2;

  /**
   * 型ごとの FFN を持たない形。
   *
   * @param hidden 隠れ状態の幅 H（32 の倍数）
   * @param ffn FFN の幅 F（128 の倍数）。バックボーンと候補層の全ての層がこの幅を使う
   * @param gate ヘッドの隠れ層の幅 G（64 の倍数。B とゲートで同じ）
   * @param depth バックボーンの層の数 L（{@link #MIN_DEPTH} から {@link #MAX_DEPTH}）
   */
  record V2(int hidden, int ffn, int gate, int depth) implements TokenTrunk {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 幅が倍数の決まりを外れるか、深さが範囲の外のとき
     */
    public V2 {
      requireWidths(hidden, ffn, gate);
      if (depth < MIN_DEPTH || depth > MAX_DEPTH) {
        throw new IllegalArgumentException("depth must be 2 or 3: " + depth);
      }
    }

    /**
     * 候補層の 2 段目を持たない。
     *
     * @return false
     */
    @Override
    public boolean reread() {
      return false;
    }
  }

  /**
   * 型ごとの FFN を持つ形。深さは {@link #DEPTH} で、層 1 から後の層では、局・席のトークンと牌のトークンが別々の FFN を通る。
   *
   * @param hidden 隠れ状態の幅 H（32 の倍数）
   * @param ffn 層 0 の FFN の幅 F（128 の倍数）
   * @param gate ヘッドの隠れ層の幅 G（64 の倍数。B とゲートで同じ）
   * @param tile 牌のトークンの FFN の幅（128 の倍数）
   * @param context 局・席のトークンの FFN の幅（128 の倍数）
   * @param candidate 候補層の FFN の幅（128 の倍数）
   * @param reread 候補層の 2 段目（遷移ごとにイベントを読み直す、Attention だけの段）を持つなら true
   */
  record V3(int hidden, int ffn, int gate, int tile, int context, int candidate, boolean reread)
      implements TokenTrunk {

    /** バックボーンの層の数。 */
    public static final int DEPTH = 3;

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 幅が倍数の決まりを外れるとき
     */
    public V3 {
      requireWidths(hidden, ffn, gate);
      requireFfn("tile", tile);
      requireFfn("context", context);
      requireFfn("candidate", candidate);
    }

    /**
     * バックボーンの層の数を返す。
     *
     * @return {@link #DEPTH}
     */
    @Override
    public int depth() {
      return DEPTH;
    }
  }

  /**
   * 隠れ状態の幅 H を返す。
   *
   * @return H
   */
  int hidden();

  /**
   * ヘッドの隠れ層の幅 G を返す。
   *
   * @return G
   */
  int gate();

  /** 埋め込みに適用するトークン層を入力側から順に返す。 */
  List<HipLayer> layers();

  private static HipLayer allTokens(int ffn) {
    return new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(ffn));
  }

  /** 専用の HIP カーネルが扱わず、eager で計算する構造なら true。 */
  default boolean eagerOnly() {
    return !HipCapabilities.supports(hidden(), candidateSegments(), reread());
  }

  /** 候補層が ΣC と K を持つなら true。 */
  default boolean candidateSegments() {
    return inputFeatures().contains(InputFeature.CANDIDATE_SEGMENTS);
  }

  /** バックボーンの Attention と正規化を {@link AttentionTape} に渡せる形なら true。 */
  default boolean supportsAttentionTape() {
    return true;
  }

  int ffn();

  int depth();

  boolean reread();

  /** 候補の段数と、読み直しのキー・バリューを取り出す位置。 */
  default CandidateStages candidateStages() {
    return reread() ? CandidateStages.REREAD : CandidateStages.ONE;
  }

  /** 局・席・牌とイベントに対して層を重ねる幹。 */
  sealed interface TokenTrunk extends PrismShape permits V2, V3, V4 {
    /**
     * バックボーンの各層の役目と FFN の幅を返す。
     *
     * @return 層（入力に近い順）
     */
    @Override
    default List<HipLayer> layers() {
      List<HipLayer> layers = new ArrayList<>();
      switch (this) {
        case V2 v2 -> {
          HipLayer allTokens = allTokens(v2.ffn());
          for (int layer = 0; layer < v2.depth() - 1; layer++) {
            layers.add(allTokens);
          }
          layers.add(new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(v2.ffn())));
        }
        case V4 v4 -> {
          layers.add(allTokens(v4.layer0Ffn()));
          for (int layer = 1; layer < v4.depth(); layer++) {
            layers.add(new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(v4.ffn())));
          }
        }
        case V3 v3 -> {
          layers.add(allTokens(v3.ffn()));
          for (int layer = 1; layer < V3.DEPTH; layer++) {
            layers.add(
                new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Typed(v3.context(), v3.tile())));
          }
        }
      }
      return List.copyOf(layers);
    }
  }

  /**
   * 層・候補層・ヘッドと入力形式を HIP の記述に変換する。
   *
   * @throws IllegalArgumentException 専用のカーネルが扱えない構造・幅・入力形式のとき
   */
  default HipArchitecture hip(InputFormat format) {
    if (eagerOnly()) {
      throw new IllegalArgumentException(
          "HIP does not support this shape; use eager inference: " + this);
    }
    int gateKinds = (int) format.roles().stream().filter(InputFormat.Role::gate).count();
    return new HipArchitecture(
        hidden(),
        (float) epsilon(),
        (float) EncoderLayer.SCALE,
        ValueGraphHead.TAU,
        layers(),
        candidateFfn(),
        candidateStages(),
        gate(),
        gateKinds,
        format.roles().size(),
        counts(format));
  }

  /** イベント同士が層 0 で読める範囲。固定トークンはどちらでも全イベントを読む。 */
  enum EventAttention {
    /** 自分と、それ以前に公開されたイベントを読む。 */
    CAUSAL,
    /** 全イベントを双方向に読む。 */
    BIDIRECTIONAL
  }

  /**
   * 全ての関係区分を持つ幹と、牌の非正規化和（ΣC）・関係表（K）を併用する一段の候補層。
   *
   * @param hidden 隠れ状態の幅。S は 256、M は 384、L は 512
   * @param layer0Ffn イベントも更新する最初の層の FFN 幅。後続の層と候補層の幅は hidden の 2 倍
   * @param depth 幹の層数（2 または 3）。層 1 以降は固定トークンだけを更新する
   * @param eventAttention イベント同士の注意の向き
   */
  record V4(int hidden, int layer0Ffn, int depth, EventAttention eventAttention)
      implements TokenTrunk {
    public V4 {
      if (HipCapabilities.width(hidden) == null) {
        throw new IllegalArgumentException("undeclared V4 hidden width " + hidden);
      }
      requireWidths(hidden, layer0Ffn, hidden);
      if (depth < MIN_DEPTH || depth > MAX_DEPTH) {
        throw new IllegalArgumentException("V4 depth must be 2 or 3: " + depth);
      }
      Objects.requireNonNull(eventAttention, "eventAttention");
    }

    /** 指定した幅で層 0 の FFN を 2 倍幅にし、深さ 2・因果注意で作る。 */
    public V4(int hidden) {
      this(hidden, 2 * hidden, DEFAULT_DEPTH, EventAttention.CAUSAL);
    }

    /** S の既定形。 */
    public V4() {
      this(HipArchitecture.HIDDEN);
    }

    @Override
    public int ffn() {
      return 2 * hidden;
    }

    @Override
    public int gate() {
      return hidden;
    }

    @Override
    public boolean reread() {
      return false;
    }

    @Override
    public EncoderLayer encoder(int layer, InputFormat format) {
      return causalEvents() && layer == 0
          ? new CausalEventLayer(this, layer, format)
          : new EncoderLayer(this, layer, format);
    }

    public boolean causalEvents() {
      return eventAttention == EventAttention.CAUSAL;
    }

    @Override
    public boolean supportsAttentionTape() {
      return !eagerOnly();
    }

    @Override
    public Set<InputFeature> inputFeatures() {
      return causalEvents()
          ? Set.of(InputFeature.CANDIDATE_SEGMENTS, InputFeature.PUBLIC_ORDER_EVENTS)
          : Set.of(InputFeature.CANDIDATE_SEGMENTS);
    }

    @Override
    public CandidateLayer.Features candidateFeatures(InputFormat format) {
      return new CoreCandidate(this, format);
    }
  }

  private static void requireWidths(int hidden, int ffn, int gate) {
    if (hidden <= 0 || hidden % (2 * HEAD_WIDTH) != 0) {
      throw new IllegalArgumentException("hidden must be a positive multiple of 32: " + hidden);
    }
    requireFfn("ffn", ffn);
    if (gate <= 0 || gate % HipArchitecture.GATE_STEP != 0) {
      throw new IllegalArgumentException(
          "gate must be a positive multiple of " + HipArchitecture.GATE_STEP + ": " + gate);
    }
  }

  private static void requireFfn(String name, int width) {
    if (width <= 0 || width % HipFfn.CHUNK != 0) {
      throw new IllegalArgumentException(
          name + " must be a positive multiple of " + HipFfn.CHUNK + ": " + width);
    }
  }

  /**
   * 系譜を始めるときの形を返す。H 256・F 512・G 256 で、深さは既定の 2。
   *
   * @return 形
   */
  static V2 standard() {
    return standard(DEFAULT_DEPTH);
  }

  /**
   * 系譜を始めるときの幅（H 256・F 512・G 256）で、深さだけを選んだ形を返す。
   *
   * @param depth バックボーンの層の数
   * @return 形
   */
  static V2 standard(int depth) {
    return new V2(256, 512, 256, depth);
  }

  /**
   * ネットワークの形式を返す。
   *
   * @return 形の型に対応する保存形式
   */
  default PrismArchitecture architecture() {
    return switch (this) {
      case V2 v2 -> PrismArchitecture.V2;
      case V3 v3 -> PrismArchitecture.V3;
      case V4 v4 -> PrismArchitecture.V4;
    };
  }

  /**
   * Attention の幅 A を返す。
   *
   * @return H/2
   */
  default int attention() {
    return hidden() / 2;
  }

  /**
   * Attention のヘッドの数を返す。
   *
   * @return H/32
   */
  default int heads() {
    return attention() / HEAD_WIDTH;
  }

  /**
   * RMSNorm の ε を返す。
   *
   * @return {@link #NORM_EPSILON}·{@link #NORM_REFERENCE_WIDTH}/H
   */
  default double epsilon() {
    return NORM_EPSILON * NORM_REFERENCE_WIDTH / hidden();
  }

  /**
   * 候補層の FFN の幅を返す。
   *
   * @return {@link V2} なら F、{@link V3} なら候補層の幅
   */
  default int candidateFfn() {
    return switch (this) {
      case V2 v2 -> v2.ffn();
      case V3 v3 -> v3.candidate();
      case V4 v4 -> v4.ffn();
    };
  }

  /**
   * 入力形式で決まる、HIP が使う数を返す。牌の数は 1 行あたりの牌のトークンの数で、牌の関係の表（{@link TileTokens#relations()}）
   * はその 2 乗の大きさになる。関係のクラスの数は {@link MahjongTileRelation} の数で、四人打ちと三人打ちで同じ。
   *
   * @param format 入力形式
   * @return 席の数、牌の数、遷移のキーのコードの数、牌の関係のクラスの数、イベントの数の上限
   * @throws IllegalArgumentException 局が 1 トークンでない形式のとき
   */
  static HipCounts counts(InputFormat format) {
    InputFormat.Keys keys = format.keys();
    if (keys.round() != 1) {
      throw new IllegalArgumentException(
          "the HIP architecture needs one round token per row, got " + keys.round());
    }
    return new HipCounts(
        keys.seat(),
        keys.tile(),
        keys.codes(),
        MahjongTileRelation.values().length,
        format.maxEvents());
  }
}
