package com.epsilon.prism.hip;

import java.util.List;
import java.util.Objects;

/**
 * ネットワークの形の記述。ネットワークの層の並び（{@code PrismShape.layers()}）、WeightPack のヘッダー、計画（{@link HipShape}）、
 * native（{@code prism_hip_format.h} の {@code Architecture}）は、どれもこの記述を読む。
 *
 * <p>バックボーンは、全トークンの層が 1 つ以上続いた後に、局・席・牌の層が 1 つ以上続く。層の数は合わせて
 * {@link #MIN_DEPTH}〜{@link #MAX_DEPTH}。全トークンの層の FFN は、トークンの型で分けない。局・席・牌の層は、イベントの残差を
 * 書き換えない。
 *
 * @param hidden 残差ストリームの幅 H（{@link #HIDDEN} だけを受け付ける）
 * @param epsilon RMSNorm で二乗の平均に足す小さな値
 * @param scale Attention のロジットに掛ける尺度
 * @param tau 集約の温度
 * @param layers バックボーンの層（入力に近い順）
 * @param candidateFfn 候補層の FFN の幅
 * @param stages 候補層の段の組み方
 * @param gate ヘッドの隠れ層の幅 G（打牌の値とゲートのヘッドで共通。{@link #GATE_STEP} の倍数）
 * @param gateKinds ゲートの種類の数
 * @param roles 価値グラフでの手の扱いの種類の数
 * @param counts 四人打ちか三人打ちかで決まる数
 */
public record HipArchitecture(
    int hidden,
    float epsilon,
    float scale,
    float tau,
    List<HipLayer> layers,
    int candidateFfn,
    CandidateStages stages,
    int gate,
    int gateKinds,
    int roles,
    HipCounts counts) {

  /** カーネルが前提にしている H。 */
  public static final int HIDDEN = 256;

  public static final int HEAD_WIDTH = 16;

  /** G はこの値の倍数にする。B とゲートの WMMA が、列をこの幅ずつ処理するため。 */
  public static final int GATE_STEP = 64;

  public static final int MIN_DEPTH = 2;

  public static final int MAX_DEPTH = 3;

  /** 手の扱いの種類の上限。K7 が持つ、扱いごとの表の大きさ。 */
  public static final int MAX_ROLES = 16;

  public static final int MIN_SEATS = 2;

  /** 席の数の上限。関係のコードのうち席を表す値は、最大で席の数の 3 倍になる。4 席までなら 4 bit に収まる。 */
  public static final int MAX_SEATS = 4;

  /** 牌の種類の上限。K0・T0 は、牌の関係の表をこの大きさで LDS に置く。 */
  public static final int MAX_TILES = 34;

  /** 1 行の局・席・牌のトークンの上限。K4・K6 は、クエリとキーのタイルをこの大きさで持つ。 */
  public static final int MAX_FIXED_TOKENS = 48;

  /** 1 行のトークンの上限。K0・K2・K4 は、1 行をこの大きさで LDS に置く。 */
  public static final int MAX_ROW_TOKENS = 160;

  /** コードの表の行の上限。遷移のコードは UINT8 で、2 段目のコードもその後ろに続く。 */
  public static final int MAX_CODES = 255;

  /** 1 層を書くのに使う int の数（層の種類、型ごとの FFN か、組ごとの FFN の幅）。 */
  private static final int LAYER_INTS = 2 + TokenGroup.values().length;

  /** native の {@code Architecture} の int の数。 */
  public static final int INTS = 5 + MAX_DEPTH * LAYER_INTS + 5 + 5;

  /**
   * 記述を確かめる。
   *
   * @throws IllegalArgumentException 決まりを外れるとき
   */
  public HipArchitecture {
    Objects.requireNonNull(stages, "stages");
    Objects.requireNonNull(counts, "counts");
    layers = List.copyOf(layers);
    if (hidden != HIDDEN) {
      throw new IllegalArgumentException("HIP kernels are specialised for hidden " + HIDDEN + " only, got " + hidden);
    }
    for (float value : new float[] {epsilon, scale, tau}) {
      if (!(value > 0f) || !Float.isFinite(value)) {
        throw new IllegalArgumentException("epsilon, scale and tau must be positive and finite: " + epsilon + ", "
            + scale + ", " + tau);
      }
    }
    if (layers.size() < MIN_DEPTH || layers.size() > MAX_DEPTH) {
      throw new IllegalArgumentException("depth must be " + MIN_DEPTH + ".." + MAX_DEPTH + ", got " + layers.size());
    }
    int allTokens = 0;
    while (allTokens < layers.size() && layers.get(allTokens).role() == LayerRole.ALL_TOKENS) {
      allTokens++;
    }
    if (allTokens == 0 || allTokens == layers.size()) {
      throw new IllegalArgumentException("the backbone needs all-token layers followed by fixed-token layers: " + layers);
    }
    for (int layer = 0; layer < layers.size(); layer++) {
      HipLayer spec = layers.get(layer);
      if (layer >= allTokens && spec.role() != LayerRole.FIXED_TOKENS) {
        throw new IllegalArgumentException("an all-token layer follows a fixed-token layer: " + layers);
      }
      if (spec.role() == LayerRole.ALL_TOKENS && !(spec.ffn() instanceof HipFfn.Shared)) {
        throw new IllegalArgumentException("all-token layer " + layer + " must have a shared FFN, got " + spec.ffn());
      }
    }
    if (candidateFfn <= 0 || candidateFfn % HipFfn.CHUNK != 0) {
      throw new IllegalArgumentException(
          "candidate ffn must be a positive multiple of " + HipFfn.CHUNK + ", got " + candidateFfn);
    }
    if (gate <= 0 || gate % GATE_STEP != 0) {
      throw new IllegalArgumentException("gate must be a positive multiple of " + GATE_STEP + ", got " + gate);
    }
    if (gateKinds <= 0 || gateKinds > roles || roles > MAX_ROLES) {
      throw new IllegalArgumentException(
          "need 0 < gate kinds <= roles <= " + MAX_ROLES + ", got " + gateKinds + " and " + roles);
    }
    if (counts.seats() < MIN_SEATS || counts.seats() > MAX_SEATS) {
      throw new IllegalArgumentException("seats must be " + MIN_SEATS + ".." + MAX_SEATS + ", got " + counts.seats());
    }
    if (counts.tiles() > MAX_TILES) {
      throw new IllegalArgumentException("tiles must be at most " + MAX_TILES + ", got " + counts.tiles());
    }
    if (counts.fixed() > MAX_FIXED_TOKENS || counts.rowTokens() > MAX_ROW_TOKENS) {
      throw new IllegalArgumentException("rows of " + counts.fixed() + " fixed and " + counts.rowTokens()
          + " tokens exceed " + MAX_FIXED_TOKENS + " and " + MAX_ROW_TOKENS);
    }
    if (counts.rereadCodes() > MAX_CODES) {
      throw new IllegalArgumentException("code tables of " + counts.rereadCodes() + " rows exceed " + MAX_CODES);
    }
  }

  /** バックボーンの層の数 L。 */
  public int depth() {
    return layers.size();
  }

  /** Attention の幅 A = H/2（ヘッドの幅 16 × ヘッドの数 H/32）。 */
  public int attention() {
    return hidden / 2;
  }

  /** ヘッドの数 H/32。 */
  public int heads() {
    return attention() / HEAD_WIDTH;
  }

  /**
   * バックボーンの層を返す。
   *
   * @param layer 層の番号（0 から）
   * @return 層
   */
  public HipLayer layer(int layer) {
    return layers.get(layer);
  }

  /** 全トークンの層の数。局・席・牌の層は、この番号から始まる。 */
  public int allTokenLayers() {
    int count = 0;
    while (layers.get(count).role() == LayerRole.ALL_TOKENS) {
      count++;
    }
    return count;
  }

  /** 局・席・牌の層の数。 */
  public int fixedLayers() {
    return depth() - allTokenLayers();
  }

  /**
   * 局・席・牌の層の中の番号を返す（最初の局・席・牌の層が 0）。
   *
   * @param layer 局・席・牌の層の番号
   * @return 局・席・牌の層の中の番号
   * @throws IllegalArgumentException 局・席・牌の層でないとき
   */
  public int fixedIndex(int layer) {
    if (layer < 0 || layer >= depth() || layers.get(layer).role() != LayerRole.FIXED_TOKENS) {
      throw new IllegalArgumentException("layer " + layer + " is not a fixed-token layer of " + layers);
    }
    return layer - allTokenLayers();
  }

  /**
   * イベントのキーとバリューを写して使う局・席・牌の層の数。写すのは、最後の全トークンの層の K3 が作ったもの。最初の局・席・牌の層
   * だけは、全てのトークンの n から自分で射影する。{@link HipRegion#EVENT_KEY_VALUE} は、層 l の分を {@code fixedIndex(l) − 1} 番目に
   * 置く。
   */
  public int copiedEventLayers() {
    return fixedLayers() - 1;
  }

  /** 候補層の 2 段目を持つなら true。 */
  public boolean reread() {
    return stages == CandidateStages.REREAD;
  }

  /** 最後の正規化がゲインを持つ組。最後の層の FFN の組と同じ。 */
  public List<TokenGroup> finalGroups() {
    return layers.getLast().ffn().groups();
  }

  /**
   * この形に、重みのバッファのその項目があるかを返す。
   *
   * @param weight 重みのバッファ
   * @param entry 項目（{@link HipWeight.Index} の番号）
   * @return あるなら true
   */
  public boolean holds(HipWeight weight, int entry) {
    return switch (weight.index()) {
      case SINGLE -> entry == 0 && (!weight.reread() || reread());
      case LAYER -> entry >= 0 && entry < depth();
      case FFN -> {
        if (entry == HipWeight.CANDIDATE_FFN) {
          yield true;
        }
        int layer = entry / TokenGroup.values().length;
        yield entry >= 0
            && layer < depth()
            && layers.get(layer).ffn().groups().contains(TokenGroup.values()[entry % TokenGroup.values().length]);
      }
      case GROUP -> entry >= 0 && entry < TokenGroup.values().length && finalGroups().contains(TokenGroup.values()[entry]);
    };
  }

  /**
   * FFN の重みの項目について、その FFN の幅を返す。
   *
   * @param entry 項目（{@link HipWeight#ffnEntry} か {@link HipWeight#CANDIDATE_FFN}）
   * @return FFN の幅
   * @throws IllegalArgumentException この形にない項目のとき
   */
  public int ffnWidth(int entry) {
    if (entry == HipWeight.CANDIDATE_FFN) {
      return candidateFfn;
    }
    int groups = TokenGroup.values().length;
    if (entry < 0 || entry / groups >= depth()) {
      throw new IllegalArgumentException("FFN entry " + entry + " is outside the " + depth() + " layers");
    }
    return layers.get(entry / groups).ffn().width(TokenGroup.values()[entry % groups]);
  }

  /**
   * native の {@code Architecture} と同じ並びの int の列を返す。float はビットのまま入れる。層の数を超えた分の層は、層の種類を −1 に
   * し、残りを 0 にする。
   *
   * @return {@link #INTS} 個の int
   */
  public int[] encode() {
    int[] ints = new int[INTS];
    int at = 0;
    ints[at++] = hidden;
    ints[at++] = Float.floatToRawIntBits(epsilon);
    ints[at++] = Float.floatToRawIntBits(scale);
    ints[at++] = Float.floatToRawIntBits(tau);
    ints[at++] = depth();
    for (int layer = 0; layer < MAX_DEPTH; layer++) {
      if (layer >= depth()) {
        ints[at] = -1;
        at += LAYER_INTS;
        continue;
      }
      HipLayer spec = layers.get(layer);
      ints[at++] = spec.role().ordinal();
      ints[at++] = spec.ffn() instanceof HipFfn.Typed ? 1 : 0;
      for (TokenGroup group : TokenGroup.values()) {
        ints[at++] = spec.ffn().groups().contains(group) ? spec.ffn().width(group) : 0;
      }
    }
    ints[at++] = candidateFfn;
    ints[at++] = stages.ordinal();
    ints[at++] = gate;
    ints[at++] = gateKinds;
    ints[at++] = roles;
    ints[at++] = counts.seats();
    ints[at++] = counts.tiles();
    ints[at++] = counts.keyCodes();
    ints[at++] = counts.tileRelations();
    ints[at++] = counts.maxEvents();
    if (at != INTS) {
      throw new IllegalStateException("encoded " + at + " ints, expected " + INTS);
    }
    return ints;
  }
}
