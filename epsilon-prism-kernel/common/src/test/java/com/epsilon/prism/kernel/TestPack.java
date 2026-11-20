package com.epsilon.prism.kernel;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * ネットワークを知らないテストで使う、形の記述と WeightPack のヘッダー。ヘッダーは形の記述から作り、入力の列のスキーマは native の
 * 照合を通る最小の値にする（どの数の記述でも作れる）。並べる重みは呼び出し元が決める。
 */
final class TestPack {

  /** 四人打ちの数（席 4・牌 34・キーのコード 109・牌の関係の種類 23・イベントの上限 110）。 */
  static final TokenCounts FOUR_PLAYERS = new TokenCounts(4, 34, 109, 23, 110);

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
  static CollectionArchitecture architecture(int ffn, TokenCounts counts) {
    return new CollectionArchitecture(
        CollectionArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(
            new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(ffn)),
            new EncoderLayerShape(LayerRole.FIXED_TOKENS, new FeedForwardShape.Shared(ffn))),
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
  static PackedWeightLayout.Header header(CollectionArchitecture architecture) {
    PackedWeightLayout.Header header =
        new PackedWeightLayout.Header(architecture).schema(ColumnIndex.RELATION_ROWS, 112);
    ColumnIndex[] rows = {
      ColumnIndex.ROW_FIRST_EVENT,
      ColumnIndex.ROW_EVENTS,
      ColumnIndex.ROW_FIRST_ACTION,
      ColumnIndex.ROW_ACTIONS,
      ColumnIndex.ROW_FIRST_TRANSITION,
      ColumnIndex.ROW_TRANSITIONS
    };
    for (int i = 0; i < rows.length; i++) {
      header.schema(rows[i], i);
    }
    ColumnIndex[] actions = {
      ColumnIndex.ACTION_ROW,
      ColumnIndex.ACTION_FIRST_TRANSITION,
      ColumnIndex.ACTION_TRANSITIONS,
      ColumnIndex.ACTION_ROLE,
      ColumnIndex.ACTION_BASE,
      ColumnIndex.ACTION_REPRESENTATIVE
    };
    for (int i = 0; i < actions.length; i++) {
      header.schema(actions[i], i);
    }
    ColumnIndex[] relation = {
      ColumnIndex.RELATION_TILE,
      ColumnIndex.RELATION_OWNER,
      ColumnIndex.RELATION_EVENT,
      ColumnIndex.RELATION_RIICHI,
      ColumnIndex.RELATION_LAST_DISCARD,
      ColumnIndex.RELATION_LINK,
      ColumnIndex.RELATION_SAFETY
    };
    for (int i = 0; i < relation.length; i++) {
      header.schema(relation[i], i);
    }
    header
        .schema(ColumnIndex.TRANSITION_CALL_DISCARD, 0)
        .schema(ColumnIndex.TRANSITION_MAIN_TILE, 1);
    for (InputSection section : InputSection.values()) {
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

  /** little-endian のダイレクトの ByteBuffer を確保する。 */
  static ByteBuffer direct(long bytes) {
    return ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
  }
}
