package com.epsilon.prism.hip;

/**
 * カーネルが読む重みの種類。並びは native と同じ。行列は BF16 で、DJL の Linear と同じ {@code [out, in]}。バイアスは FP32 で持つが、
 * 値は eager の Linear に合わせて BF16 に丸めてある。ゲイン・関係のバイアス・コードの表・ヘッドの最後の層は、FP32 のまま持つ。
 *
 * <p>重みごとに引き方（{@link Index}）があり、引き方の項目ごとに 1 つの位置を WeightPack のヘッダーに持つ。埋め込みの表と型ごとの
 * Linear は、組ごとに折り込んだ表として {@link HipEmbedding} が持つ。
 */
public enum HipWeight {
  /** 牌どうしの関係の種類 {@code [牌·牌]}（UINT8。MahjongTileRelation の ordinal）。 */
  TILE_RELATION(HipElement.U8, Index.SINGLE),
  /** Attention の前の RMSNorm のゲイン {@code [H]}。 */
  ATTENTION_NORM(HipElement.FP32, Index.LAYER),
  /** q・k・v の射影 {@code [3A, H]}（行は q・k・v の順で、それぞれの中はヘッドの順）。 */
  QKV(HipElement.BF16, Index.LAYER),
  /** q・k・v のバイアス {@code [3A]}。 */
  QKV_BIAS(HipElement.FP32, Index.LAYER),
  /** 関係のバイアスの表 {@code [112, ヘッド]}。 */
  RELATION_BIAS(HipElement.FP32, Index.LAYER),
  /** O 射影 {@code [H, A]}。 */
  OUTPUT(HipElement.BF16, Index.LAYER),
  /** O 射影のバイアス {@code [H]}。 */
  OUTPUT_BIAS(HipElement.FP32, Index.LAYER),
  /** FFN の前の RMSNorm のゲイン {@code [H]}（バックボーンの層と組の対ごとに 1 つと、候補層に 1 つ）。 */
  FFN_NORM(HipElement.FP32, Index.FFN),
  /** FFN の上り射影 {@code [F, H]}（F はその項目の FFN の幅）。 */
  UP(HipElement.BF16, Index.FFN),
  /** 上り射影のバイアス {@code [F]}。 */
  UP_BIAS(HipElement.FP32, Index.FFN),
  /** FFN の下り射影 {@code [H, F]}。 */
  DOWN(HipElement.BF16, Index.FFN),
  /** 下り射影のバイアス {@code [H]}。 */
  DOWN_BIAS(HipElement.FP32, Index.FFN),
  /** 最後の RMSNorm のゲイン {@code [H]}（組ごと）。 */
  FINAL_NORM(HipElement.FP32, Index.GROUP),
  /** 候補の牌の項 {@code [H, H]}（バイアスなし）。 */
  TILE_ENTRY(HipElement.BF16, Index.SINGLE),
  /** 候補のクエリの前の RMSNorm のゲイン {@code [H]}。 */
  QUERY_NORM(HipElement.FP32, Index.SINGLE),
  /** 候補のクエリの射影 {@code [A, H]}。 */
  QUERY(HipElement.BF16, Index.SINGLE),
  /** 候補のクエリのバイアス {@code [A]}。 */
  QUERY_BIAS(HipElement.FP32, Index.SINGLE),
  /** 候補の Cross-Attention のキーとバリューの射影 {@code [2A, H]}。 */
  KEY_VALUE(HipElement.BF16, Index.SINGLE),
  /** 候補の Cross-Attention のキーとバリューのバイアス {@code [2A]}。 */
  KEY_VALUE_BIAS(HipElement.FP32, Index.SINGLE),
  /** キーに付けるコードごとに、キーに足す差分 {@code [コード, A]}。 */
  CODE_KEY(HipElement.FP32, Index.SINGLE),
  /** キーに付けるコードごとに、バリューに足す差分 {@code [コード, A]}。 */
  CODE_VALUE(HipElement.FP32, Index.SINGLE),
  /** 候補の O 射影 {@code [H, A]}。 */
  CANDIDATE_OUTPUT(HipElement.BF16, Index.SINGLE),
  /** 候補の O 射影のバイアス {@code [H]}。 */
  CANDIDATE_OUTPUT_BIAS(HipElement.FP32, Index.SINGLE),
  /** 2 段目のクエリの前の RMSNorm のゲイン {@code [H]}。 */
  REREAD_QUERY_NORM(HipElement.FP32, Index.SINGLE),
  /** 2 段目のクエリの射影 {@code [A, H]}。 */
  REREAD_QUERY(HipElement.BF16, Index.SINGLE),
  /** 2 段目のクエリのバイアス {@code [A]}。 */
  REREAD_QUERY_BIAS(HipElement.FP32, Index.SINGLE),
  /** 2 段目がイベントのキーとバリューを作る前にかける RMSNorm のゲイン {@code [H]}。 */
  REREAD_EVENT_NORM(HipElement.FP32, Index.SINGLE),
  /** 2 段目のキーとバリューの射影 {@code [2A, H]}（局・席・牌とイベントで共通）。 */
  REREAD_KEY_VALUE(HipElement.BF16, Index.SINGLE),
  /** 2 段目のキーとバリューのバイアス {@code [2A]}。 */
  REREAD_KEY_VALUE_BIAS(HipElement.FP32, Index.SINGLE),
  /** 2 段目のコードごとに、キーに足す差分 {@code [2 段目のコード, A]}。 */
  REREAD_CODE_KEY(HipElement.FP32, Index.SINGLE),
  /** 2 段目のコードごとに、バリューに足す差分 {@code [2 段目のコード, A]}。 */
  REREAD_CODE_VALUE(HipElement.FP32, Index.SINGLE),
  /** 2 段目の出力の射影 {@code [H, A]}。 */
  REREAD_OUTPUT(HipElement.BF16, Index.SINGLE),
  /** 2 段目の出力の射影のバイアス {@code [H]}。 */
  REREAD_OUTPUT_BIAS(HipElement.FP32, Index.SINGLE),
  /** 打牌の値のヘッドの RMSNorm のゲイン {@code [H]}。 */
  DISCARD_NORM(HipElement.FP32, Index.SINGLE),
  /** 打牌の値のヘッドの隠れ層 {@code [G, H]}。 */
  DISCARD_HIDDEN(HipElement.BF16, Index.SINGLE),
  /** 打牌の値のヘッドの隠れ層のバイアス {@code [G]}。 */
  DISCARD_HIDDEN_BIAS(HipElement.FP32, Index.SINGLE),
  /** 打牌の値のヘッドの最後の層 {@code [G]}。 */
  DISCARD_OUTPUT(HipElement.FP32, Index.SINGLE),
  /** 打牌の値のヘッドが FP32 のまま読むフィールドの係数 {@code [フィールド]}。 */
  DISCARD_EXACT(HipElement.FP32, Index.SINGLE),
  /** ゲートのヘッドの RMSNorm のゲイン {@code [H]}（手の表現と基準の手の表現で共通）。 */
  GATE_NORM(HipElement.FP32, Index.SINGLE),
  /** ゲートの隠れ層 {@code [G, 3H]}。 */
  GATE_HIDDEN(HipElement.BF16, Index.SINGLE),
  /** ゲートの隠れ層のバイアス {@code [G]}。 */
  GATE_HIDDEN_BIAS(HipElement.FP32, Index.SINGLE),
  /** ゲートの種類ごとのバイアス {@code [種類, G]}。 */
  GATE_KIND_OFFSET(HipElement.FP32, Index.SINGLE),
  /** ゲートが FP32 のまま読むフィールドの射影 {@code [種類, G, フィールド]}。 */
  GATE_EXACT_PROJECTION(HipElement.FP32, Index.SINGLE),
  /** ゲートの最後の層 {@code [種類, G]}。 */
  GATE_OUTPUT(HipElement.FP32, Index.SINGLE),
  /** ゲートの最後の層のバイアス {@code [種類]}。 */
  GATE_OUTPUT_BIAS(HipElement.FP32, Index.SINGLE),
  /** ゲートが FP32 のまま読むフィールドの係数 {@code [種類, フィールド]}。 */
  GATE_EXACT_OUTPUT(HipElement.FP32, Index.SINGLE);

  /** 重みの引き方。並びは native の WeightIndex と同じ。 */
  public enum Index {
    /** 項目は 1 つ。 */
    SINGLE(1),
    /** バックボーンの層ごと（項目は層の番号）。 */
    LAYER(HipArchitecture.MAX_DEPTH),
    /** バックボーンの層と組の対ごと、それに候補層（項目は {@link #ffnEntry} か {@link #CANDIDATE_FFN}）。 */
    FFN(HipArchitecture.MAX_DEPTH * TokenGroup.values().length + 1),
    /** 組ごと（項目は組の ordinal）。 */
    GROUP(TokenGroup.values().length);

    private final int entries;

    Index(int entries) {
      this.entries = entries;
    }

    /** 項目の数。 */
    public int entries() {
      return entries;
    }
  }

  /** 候補層の FFN の項目。 */
  public static final int CANDIDATE_FFN = HipArchitecture.MAX_DEPTH * TokenGroup.values().length;

  /** 全ての重みの項目の数（WeightPack のヘッダーにある位置の表の長さ）。 */
  public static final int ENTRIES = firstEntry(values().length);

  private final HipElement element;
  private final Index index;

  HipWeight(HipElement element, Index index) {
    this.element = element;
    this.index = index;
  }

  public HipElement element() {
    return element;
  }

  public Index index() {
    return index;
  }

  /** 候補層の 2 段目の重みなら true。2 段目のない形は、この重みを持たない。 */
  public boolean reread() {
    return name().startsWith("REREAD_");
  }

  /**
   * バックボーンの層と組に当たる FFN の項目を返す。
   *
   * @param layer 層の番号
   * @param group 組
   * @return 項目
   */
  public static int ffnEntry(int layer, TokenGroup group) {
    return layer * TokenGroup.values().length + group.ordinal();
  }

  /**
   * WeightPack のヘッダーにある位置の表で、この重みの最初の項目が何番目かを返す。
   *
   * @return 位置
   */
  public int firstEntry() {
    return firstEntry(ordinal());
  }

  private static int firstEntry(int ordinal) {
    int first = 0;
    for (int w = 0; w < ordinal; w++) {
      first += values()[w].index.entries();
    }
    return first;
  }
}
