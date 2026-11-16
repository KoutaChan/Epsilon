package com.epsilon.prism.model;

import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.hip.CandidateStages;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.hip.HipLayer;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.TileTokens;
import java.util.ArrayList;
import java.util.List;

/**
 * ネットワークの形。形の型がネットワークの形式（{@link PrismArchitecture}）を決める。型ごとの FFN を持たない形は {@link V2}、持つ形は
 * {@link V3}。model.json に保存し、開くときに保存した形と照合する。
 *
 * <p>Attention の幅 A = H/2（1 ヘッド 16 次元で H/32 ヘッド）と、RMSNorm の ε = 1e-5·256/H は、幅 H から決まる。
 *
 * <p>バックボーンの各層の役目と FFN の幅は {@link #layers()} が決める。{@link V2} では、最後の層だけが局・席・牌のトークンをクエリにし、
 * ほかの層は全てのトークンをクエリにする。FFN は全ての層と候補層で同じ幅 F を使う。{@link V3} では、層 0 だけが全てのトークンをクエリに
 * してイベントのトークンも書き換える（FFN の幅は F）。層 1 から後は局・席・牌のトークンだけをクエリにし、イベントのトークンは層 0 の出力の
 * ままキーとバリューとして読む。
 */
public sealed interface PrismShape permits PrismShape.V2, PrismShape.V3 {

  int HEAD_WIDTH = 16;

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
  record V2(int hidden, int ffn, int gate, int depth) implements PrismShape {

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
      implements PrismShape {

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
   * FFN の幅 F を返す。
   *
   * @return F
   */
  int ffn();

  /**
   * ヘッドの隠れ層の幅 G を返す。
   *
   * @return G
   */
  int gate();

  /**
   * バックボーンの層の数 L を返す。
   *
   * @return L
   */
  int depth();

  /**
   * 候補層の 2 段目を持つかを返す。
   *
   * @return 持つなら true
   */
  boolean reread();

  private static void requireWidths(int hidden, int ffn, int gate) {
    if (hidden <= 0 || hidden % (2 * HEAD_WIDTH) != 0) {
      throw new IllegalArgumentException("hidden must be a positive multiple of 32: " + hidden);
    }
    requireFfn("ffn", ffn);
    if (gate <= 0 || gate % 64 != 0) {
      throw new IllegalArgumentException("gate must be a positive multiple of 64: " + gate);
    }
  }

  private static void requireFfn(String name, int width) {
    if (width <= 0 || width % 128 != 0) {
      throw new IllegalArgumentException(name + " must be a positive multiple of 128: " + width);
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
   * @return {@link V2} なら {@link PrismArchitecture#V2}、{@link V3} なら {@link PrismArchitecture#V3}
   */
  default PrismArchitecture architecture() {
    return switch (this) {
      case V2 v2 -> PrismArchitecture.V2;
      case V3 v3 -> PrismArchitecture.V3;
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
   * @return 1e-5·256/H
   */
  default double epsilon() {
    return 1e-5 * 256 / hidden();
  }

  /**
   * バックボーンの各層の役目と FFN の幅を返す。
   *
   * @return 層（入力に近い順）
   */
  default List<HipLayer> layers() {
    List<HipLayer> layers = new ArrayList<>();
    HipLayer allTokens = new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(ffn()));
    switch (this) {
      case V2 v2 -> {
        for (int layer = 0; layer < v2.depth() - 1; layer++) {
          layers.add(allTokens);
        }
        layers.add(new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(v2.ffn())));
      }
      case V3 v3 -> {
        layers.add(allTokens);
        for (int layer = 1; layer < V3.DEPTH; layer++) {
          layers.add(new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Typed(v3.context(), v3.tile())));
        }
      }
    }
    return List.copyOf(layers);
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
    };
  }

  /**
   * HIP 用の形の記述を返す。層・候補層・ヘッドはこの形から作り、トークンなどの数は入力形式から作る。
   *
   * @param format 入力形式
   * @return 記述
   * @throws IllegalArgumentException HIP のカーネルが特化していない幅か、局が 1 トークンでない形式のとき
   */
  default HipArchitecture hip(InputFormat format) {
    int gateKinds = (int) format.roles().stream().filter(InputFormat.Role::gate).count();
    return new HipArchitecture(
        hidden(),
        (float) epsilon(),
        (float) EncoderLayer.SCALE,
        ValueGraphHead.TAU,
        layers(),
        candidateFfn(),
        reread() ? CandidateStages.REREAD : CandidateStages.ONE,
        gate(),
        gateKinds,
        format.roles().size(),
        counts(format));
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
      throw new IllegalArgumentException("the HIP architecture needs one round token per row, got " + keys.round());
    }
    return new HipCounts(
        keys.seat(), keys.tile(), keys.codes(), MahjongTileRelation.values().length, format.maxEvents());
  }
}
