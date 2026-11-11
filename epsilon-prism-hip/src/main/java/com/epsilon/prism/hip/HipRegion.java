package com.epsilon.prism.hip;

/**
 * 計画の作業領域を区切った領域。並びは native と同じ。カーネルとカーネルの間でグローバルメモリに残す値だけを置く。トークンを持つ領域では、
 * トークンの並びは入力のトークンの軸と同じ（局・席・牌・イベントの順）。
 */
public enum HipRegion {
  /** 入力の blob（UINT8）。forward はホストからここへ 1 回だけ写す（H2D）。 */
  INPUT(HipElement.U8),
  /**
   * 残差ストリーム {@code [トークン, H]}（BF16）。K1 が書き、K3 が上書きする。最後の全トークンの層の K3 と、局・席・牌の層の K3・K5 は、
   * 先頭の局・席・牌の分だけを書く。最後の全トークンの層より後には、イベントの残差を読む所がないので書かない。
   */
  RESIDUAL(HipElement.BF16),
  /**
   * 次の層の Attention の入力 {@code [トークン, H]}（BF16。残差ストリームに RMSNorm をかけたもの）。K3 が次の層の分を書く。
   * 全トークンの層の K3 は全てのトークンを、局・席・牌の層の K3 は局・席・牌だけを書く。K2（層 1 から）と K4 が、行列積の A の
   * フラグメントとして読む。
   */
  NORMALIZED(HipElement.BF16),
  /**
   * 層 0 の Attention の q・k・v {@code [トークン, 3A]}（BF16。列はヘッドの対の並びで、{@link HipEmbedding#foldColumn} のとおり）。K1 が、
   * 層 0 の QKV まで折り込んだ表から書く。K2 の層 0 は射影をせずにそのまま読む。
   */
  QKV0(HipElement.BF16),
  /** Attention の出力 {@code [トークン, A]}（BF16）。K2・K4 が書き、K3・K5 が読む。 */
  ATTENTION(HipElement.BF16),
  /**
   * 関係のコード {@code [行, タイル, タイル, 256]}（7 つの区分を 1 ワードに詰めた INT32）。タイルは 16 トークンで、1 行に
   * {@link HipCounts#codeBlocks()} 個ある。タイルの対ごとに 16 × 16 ワード。K0 が書き、K2・K4 が読む。
   */
  RELATION_CODE(HipElement.I32),
  /**
   * 最後の RMSNorm をかけた後の局・席・牌のトークン {@code [局・席・牌のトークン, H]}（BF16。並びはトークンの軸の先頭の
   * 局・席・牌と同じ）。K5 が書き、K6（牌）と K7（局）が読む。
   */
  STATE(HipElement.BF16),
  /** 候補層の Cross-Attention のキーとバリュー {@code [行, 局・席・牌, 2A]}（BF16。行ごとに局・席・牌の順）。K5 が書き、K6 が読む。 */
  MEMORY(HipElement.BF16),
  /** 遷移の表現のうち、遷移のフィールドだけから線形に作る部分 {@code [遷移, H]}（BF16）。K1 が書き、K6 が読む。 */
  STATIC(HipElement.BF16),
  /**
   * 遷移の表現 {@code [遷移, H]}（BF16）。K6 が書き、K7 が読む。2 段目がある形では、K6 が 1 段目の表現を書き、K9 がそれを
   * 2 段目の表現で上書きする。
   */
  TRANSITION(HipElement.BF16),
  /** 打牌の値 {@code [遷移]}（FP32）。K6 が書き（2 段目がある形では K9）、K7 が読む。 */
  VALUE(HipElement.FP32),
  /** 手のスコア {@code [手]}（FP32。要求の順）。K7 が書き、forward はここからホストへ 1 回だけ写す（D2H）。 */
  SCORES(HipElement.FP32),
  /**
   * 2 つ目以降の局・席・牌の層が読む、イベントのキーとバリュー {@code [層, イベント, 2A]}（BF16。イベントごとに [k(A) | v(A)]）。
   * 層はバッチのイベントの数ずつ続き、層 l の分は {@code fixedIndex(l) − 1} 番目に置く。最後の全トークンの層の K3 がイベントの残差から
   * 書き、K4 は自分の層の分を写して使う。最初の局・席・牌の層の K4 は RMSNorm をかけた入力から自分で射影するので、その分はここに置かない。
   */
  EVENT_KEY_VALUE(HipElement.BF16),
  /**
   * 候補層の 2 段目が読むキーとバリュー（BF16。2 段目がある形だけ）。先頭に局・席・牌の {@code [行, 局・席・牌, 2A]}（行ごと）、続けてイベントの
   * {@code [イベント, 2A]} を置く。局・席・牌の分は K8 が最後の RMSNorm をかけた後の局・席・牌のトークンから、イベントの分は
   * 最後の全トークンの層の K3 が書き、K9 が読む。
   */
  REREAD_MEMORY(HipElement.BF16);

  private final HipElement element;

  HipRegion(HipElement element) {
    this.element = element;
  }

  public HipElement element() {
    return element;
  }

  /** 計画の容量いっぱいのときのバイト数。 */
  public long capacityBytes(HipShape shape) {
    HipShape.Capacity capacity = shape.capacity();
    return bytes(shape, capacity.rows(), capacity.events(), capacity.transitions(), capacity.actions(),
        capacity.inputBytes());
  }

  /** バッチが使うバイト数（領域の先頭から数える）。 */
  public long bytes(HipShape shape, HipBatch batch) {
    return bytes(shape, batch.rows(), batch.events(), batch.transitions(), batch.actions(), batch.bytes());
  }

  private long bytes(HipShape shape, int rows, int events, int transitions, int actions, long input) {
    HipArchitecture architecture = shape.architecture();
    long hidden = architecture.hidden();
    long attention = architecture.attention();
    long fixed = (long) shape.fixedTokens() * rows;
    long tokens = fixed + events;
    long pairs = (long) architecture.counts().codeBlocks() * HipCounts.CODE_BLOCK;
    return switch (this) {
      case INPUT -> input;
      case RESIDUAL, NORMALIZED -> tokens * hidden * element.bytes();
      case QKV0 -> tokens * 3 * attention * element.bytes();
      case ATTENTION -> tokens * attention * element.bytes();
      case RELATION_CODE -> rows * pairs * pairs * element.bytes();
      case STATE -> fixed * hidden * element.bytes();
      case MEMORY -> fixed * 2 * attention * element.bytes();
      case STATIC, TRANSITION -> transitions * hidden * element.bytes();
      case VALUE -> (long) transitions * element.bytes();
      case SCORES -> (long) actions * element.bytes();
      case EVENT_KEY_VALUE -> architecture.copiedEventLayers() * events * 2 * attention * element.bytes();
      case REREAD_MEMORY -> architecture.reread() ? tokens * 2 * attention * element.bytes() : 0;
    };
  }
}
