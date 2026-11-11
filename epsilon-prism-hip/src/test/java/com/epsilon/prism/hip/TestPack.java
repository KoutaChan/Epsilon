package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * ネットワークを知らないテスト（K3 だけを回すテストなど）で使う、WeightPack のヘッダーとバッチ。ヘッダーは形の記述から作り、入力の列の
 * スキーマは native の照合を通る最小の値にする（どの数の記述でも作れる）。並べる重みは呼び出し元が決める。バッチは 1 行で、イベントの
 * 数でトークンの数を決める。K3 は入力の blob を読まないので、blob のセクションの位置は全て 0 にする。
 */
final class TestPack {

  /** 四人打ちの数（席 4・牌 34・キーのコード 109・牌の関係の種類 23・イベントの上限 110）。 */
  static final HipCounts FOUR_PLAYERS = new HipCounts(4, 34, 109, 23, 110);

  /** 関係の属性の列の数。牌・持ち主・イベント・立直・最後の打牌・つながりの 6 列の後に、相手 3 人ぶんの安全の列が続く。 */
  private static final int RELATION_WIDTH = 9;

  private TestPack() {}

  /**
   * v2 の形の記述を返す。全トークンの層 1 つの後に局・席・牌の層が 1 つ続き、G は 256、ゲートの種類は 6、手の扱いの種類は 8。
   *
   * @param ffn FFN と候補層の FFN の幅（全ての層で同じ）
   * @param counts 四人打ちか三人打ちかで決まる数
   * @return 形の記述
   */
  static HipArchitecture architecture(int ffn, HipCounts counts) {
    return new HipArchitecture(
        HipArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(
            new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(ffn)),
            new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(ffn))),
        ffn,
        CandidateStages.ONE,
        256,
        6,
        8,
        counts);
  }

  /**
   * native の照合を通る最小のヘッダーを作る。埋め込みと重みの項目は入れない。
   *
   * @param architecture 形の記述
   * @return ヘッダー
   */
  static HipPack.Header header(HipArchitecture architecture) {
    HipPack.Header header = new HipPack.Header(architecture).schema(HipSchema.RELATION_ROWS, 112);
    HipSchema[] rows = {
      HipSchema.ROW_FIRST_EVENT,
      HipSchema.ROW_EVENTS,
      HipSchema.ROW_FIRST_ACTION,
      HipSchema.ROW_ACTIONS,
      HipSchema.ROW_FIRST_TRANSITION,
      HipSchema.ROW_TRANSITIONS
    };
    for (int i = 0; i < rows.length; i++) {
      header.schema(rows[i], i);
    }
    HipSchema[] actions = {
      HipSchema.ACTION_ROW,
      HipSchema.ACTION_FIRST_TRANSITION,
      HipSchema.ACTION_TRANSITIONS,
      HipSchema.ACTION_ROLE,
      HipSchema.ACTION_BASE,
      HipSchema.ACTION_REPRESENTATIVE
    };
    for (int i = 0; i < actions.length; i++) {
      header.schema(actions[i], i);
    }
    HipSchema[] relation = {
      HipSchema.RELATION_TILE,
      HipSchema.RELATION_OWNER,
      HipSchema.RELATION_EVENT,
      HipSchema.RELATION_RIICHI,
      HipSchema.RELATION_LAST_DISCARD,
      HipSchema.RELATION_LINK,
      HipSchema.RELATION_SAFETY
    };
    for (int i = 0; i < relation.length; i++) {
      header.schema(relation[i], i);
    }
    header.schema(HipSchema.TRANSITION_CALL_DISCARD, 0).schema(HipSchema.TRANSITION_MAIN_TILE, 1);
    for (HipSection section : HipSection.values()) {
      int width =
          switch (section) {
            case RELATION -> RELATION_WIDTH;
            case ROW_COLUMN -> 7;
            case ACTION_COLUMN -> 6;
            case TRANSITION_CATEGORY -> 2;
            case TRANSITION_CODE -> architecture.counts().fixed();
            case DISCARD_EXACT -> 8;
            case ACTION_EXACT -> 10;
            default -> 1;
          };
      header.sectionWidth(section, width);
    }
    return header;
  }

  /**
   * 指定した数のトークンを入れる、1 行のバッチを作る。blob のセクションの位置は全て 0。
   *
   * @param architecture 形の記述
   * @param tokens トークンの数（局・席・牌の数以上）
   * @return バッチ
   */
  static HipBatch batch(HipArchitecture architecture, int tokens) {
    int fixed = architecture.counts().fixed();
    if (tokens < fixed) {
      throw new IllegalArgumentException("a batch row has at least " + fixed + " tokens");
    }
    return new HipBatch(1, tokens - fixed, 1, 1, inputBytes(tokens), new long[HipSection.values().length]);
  }

  /** 指定した数のトークンのバッチに要る、blob のバイト数。いちばん大きい関係の属性のセクションに合わせる。 */
  static long inputBytes(int tokens) {
    return HipPack.align((long) tokens * RELATION_WIDTH + 256);
  }

  /** little-endian のダイレクトの ByteBuffer を確保する。 */
  static ByteBuffer direct(long bytes) {
    return ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
  }
}
