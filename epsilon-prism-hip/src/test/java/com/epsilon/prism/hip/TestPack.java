package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 網を知らない試験（K3 だけを回す試験）の WeightPack と塊。入力の約束は native の照らしを通る最小の値（PRISM と同じ数の局・席・牌）
 * にし、置く重みは呼び手が決める。塊は 1 行で、出来事の数でトークンの数を決める（区画は全て位置 0。K3 は入力の blob を読まない）。
 */
final class TestPack {

  /** 行ごとの局・席・牌。 */
  static final int FIXED = 39;

  private TestPack() {}

  /** 照らしを通る最小の頭（埋め込みと重みはなし）。 */
  static HipPack.Header header() {
    HipPack.Header header =
        new HipPack.Header()
            .schema(HipSchema.SEATS, 4)
            .schema(HipSchema.TILES, 34)
            .schema(HipSchema.CODES, 109)
            .schema(HipSchema.RELATION_ROWS, 112)
            .schema(HipSchema.GATE_KINDS, 6)
            .schema(HipSchema.ROLES, 8)
            .schema(HipSchema.TAU_BITS, Float.floatToIntBits(0.01f))
            .schema(HipSchema.SCALE_BITS, Float.floatToIntBits(0.25f));
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
    header.schema(HipSchema.RELATION_SAFETIES, 3);
    header.schema(HipSchema.TRANSITION_CALL_DISCARD, 0).schema(HipSchema.TRANSITION_MAIN_TILE, 1);
    for (HipSection section : HipSection.values()) {
      int width =
          switch (section) {
            case RELATION -> 9;
            case ROW_COLUMN -> 7;
            case ACTION_COLUMN -> 6;
            case TRANSITION_CATEGORY -> 2;
            case TRANSITION_CODE -> FIXED;
            case DISCARD_EXACT -> 8;
            case ACTION_EXACT -> 10;
            default -> 1;
          };
      header.sectionWidth(section, width);
    }
    return header;
  }

  /** トークンの数のための 1 行の塊（区画は全て位置 0）。 */
  static HipBatch batch(int tokens) {
    if (tokens < FIXED) {
      throw new IllegalArgumentException("a batch row has at least " + FIXED + " tokens");
    }
    return new HipBatch(1, tokens - FIXED, 1, 1, inputBytes(tokens), new long[HipSection.values().length]);
  }

  /** トークンの数の塊が要る blob のバイト数（関係の属性の区画が最大）。 */
  static long inputBytes(int tokens) {
    return HipPack.align((long) tokens * 9 + 256);
  }

  /** 直の little-endian の ByteBuffer。 */
  static ByteBuffer direct(long bytes) {
    return ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
  }
}
