package com.epsilon.prism.kernel;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 形の記述の決まり、重みの項目、native に渡す int の並び、層の種類の並び、数だけで決まる重みの大きさ、WeightPack のヘッダーの並びを
 * 確かめる。
 */
public class CollectionLayoutTest {

  private static final TokenCounts THREE_PLAYERS = new TokenCounts(3, 27, 107, 23, 91);

  private static CollectionArchitecture typed(
      TokenCounts counts, CandidateStages stages, int gateKinds, int roles) {
    return new CollectionArchitecture(
        CollectionArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(
            new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(512)),
            new EncoderLayerShape(LayerRole.FIXED_TOKENS, new FeedForwardShape.Typed(1536, 768)),
            new EncoderLayerShape(LayerRole.FIXED_TOKENS, new FeedForwardShape.Typed(1536, 768))),
        1024,
        stages,
        512,
        gateKinds,
        roles,
        counts);
  }

  @Test
  public void countsFollowTheSeatsTilesAndEvents() {
    assertEquals(TestPack.FOUR_PLAYERS.fixed(), 39);
    assertEquals(TestPack.FOUR_PLAYERS.rowTokens(), 149);
    assertEquals(TestPack.FOUR_PLAYERS.codeBlocks(), 10);
    assertEquals(TestPack.FOUR_PLAYERS.rereadCodes(), 133);
    assertEquals(TestPack.FOUR_PLAYERS.opponents(), 3);
    assertEquals(THREE_PLAYERS.fixed(), 31);
    assertEquals(THREE_PLAYERS.rowTokens(), 122);
    assertEquals(THREE_PLAYERS.codeBlocks(), 8);
    assertEquals(THREE_PLAYERS.rereadCodes(), 131);
    assertEquals(THREE_PLAYERS.opponents(), 2);
  }

  /**
   * 重みの項目は、層の種類・トークンの組・候補層の段で決まる。型ごとの FFN の層は組ごとに FFN を持ち、最後の正規化は最後の層の組ごとに
   * 持つ。2 段目の重みは、2 段目のある形だけが持つ。
   */
  @Test
  public void entriesFollowTheLayers() {
    CollectionArchitecture v2 = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
    CollectionArchitecture v3 = typed(TestPack.FOUR_PLAYERS, CandidateStages.REREAD, 6, 8);
    assertTrue(v2.holds(WeightField.UP, WeightField.ffnEntry(1, TokenGroup.ALL)));
    assertFalse(v2.holds(WeightField.UP, WeightField.ffnEntry(1, TokenGroup.TILE)));
    assertFalse(v2.holds(WeightField.QKV, 2));
    assertFalse(v2.holds(WeightField.REREAD_OUTPUT, 0));
    assertTrue(v2.holds(WeightField.FINAL_NORM, TokenGroup.ALL.ordinal()));
    assertTrue(v3.holds(WeightField.UP, WeightField.ffnEntry(2, TokenGroup.TILE)));
    assertFalse(v3.holds(WeightField.UP, WeightField.ffnEntry(1, TokenGroup.ALL)));
    assertTrue(v3.holds(WeightField.UP, WeightField.CANDIDATE_FFN));
    assertTrue(v3.holds(WeightField.REREAD_OUTPUT, 0));
    assertFalse(v3.holds(WeightField.FINAL_NORM, TokenGroup.ALL.ordinal()));
    assertTrue(v3.holds(WeightField.FINAL_NORM, TokenGroup.CONTEXT.ordinal()));
    assertEquals(v3.ffnWidth(WeightField.ffnEntry(1, TokenGroup.CONTEXT)), 1536);
    assertEquals(v3.ffnWidth(WeightField.CANDIDATE_FFN), 1024);
    assertEquals(v3.allTokenLayers(), 1);
    assertEquals(v3.fixedIndex(2), 1);
  }

  /**
   * native に渡す並び: 先頭の 5 つ（H・ε・尺度・τ・深さ）、層ごとの種類・FFN の型・組ごとの幅（深さより後ろの層は種類が −1）、候補層と
   * ヘッドの 5 つ、四人打ちか三人打ちかで決まる 5 つの数。
   */
  @Test
  public void encodesInNativeOrder() {
    int[] ints = typed(THREE_PLAYERS, CandidateStages.ONE, 7, 9).encode();
    assertEquals(ints.length, CollectionArchitecture.INTS);
    assertEquals(ints[0], 256);
    assertEquals(ints[1], Float.floatToRawIntBits(1e-5f));
    assertEquals(ints[4], 3);
    assertEquals(ints[5], LayerRole.ALL_TOKENS.ordinal());
    assertEquals(ints[7], 512);
    assertEquals(ints[10], LayerRole.FIXED_TOKENS.ordinal());
    assertEquals(ints[11], 1);
    assertEquals(ints[12], 0);
    assertEquals(ints[13], 1536);
    assertEquals(ints[14], 768);
    assertEquals(ints[20], 1024);
    assertEquals(ints[21], CandidateStages.ONE.ordinal());
    assertEquals(ints[23], 7);
    assertEquals(ints[24], 9);
    assertEquals(ints[25], 3);
    assertEquals(ints[29], 91);
    int[] shallow = TestPack.architecture(512, TestPack.FOUR_PLAYERS).encode();
    assertEquals(shallow[15], -1);
    assertEquals(shallow[18], 0);
  }

  /** 全トークンの層が先に並び、局・席・牌の層が後に続く。2 段目の候補層は、2 段目のある形だけが持つ。 */
  @Test
  public void layersSplitIntoAllTokenAndFixedTokenLayers() {
    CollectionArchitecture v2 = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
    EncoderLayerShape all =
        new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(512));
    CollectionArchitecture deep =
        layers(
            List.of(
                all,
                all,
                new EncoderLayerShape(LayerRole.FIXED_TOKENS, new FeedForwardShape.Shared(512))));
    CollectionArchitecture v3 = typed(TestPack.FOUR_PLAYERS, CandidateStages.REREAD, 6, 8);
    CollectionArchitecture oneStage = typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 6, 8);
    assertEquals(v2.allTokenLayers(), 1);
    assertEquals(v2.fixedLayers(), 1);
    assertEquals(v2.fixedIndex(1), 0);
    assertEquals(deep.allTokenLayers(), 2);
    assertEquals(deep.fixedLayers(), 1);
    assertEquals(deep.fixedIndex(2), 0);
    assertThrows(IllegalArgumentException.class, () -> deep.fixedIndex(1));
    assertEquals(v3.allTokenLayers(), 1);
    assertEquals(v3.fixedLayers(), 2);
    assertEquals(v3.fixedIndex(2), 1);
    assertTrue(v3.reread());
    assertFalse(oneStage.reread());
    assertFalse(v2.reread());
  }

  /**
   * 重みの大きさは、形の記述の数だけから決まる（四人打ちと、席 3・牌 27・遷移のキーのコード 107・イベント 91・ゲートの種類 7・
   * 手の扱いの種類 9 の三人打ちで確かめる）。牌の関係の表は牌の数の 2 乗、コードの表はキーのコードと 2 段目のコードの数、ゲートの表は
   * ゲートの種類の数で決まる。イベントのキーとバリューを写す層は、2 つ目からの局・席・牌の層の数だけある。
   */
  @Test
  public void sizesFollowTheCounts() {
    for (boolean three : new boolean[] {false, true}) {
      TokenCounts counts = three ? THREE_PLAYERS : TestPack.FOUR_PLAYERS;
      int gateKinds = three ? 7 : 6;
      int roles = three ? 9 : 8;
      CollectionArchitecture architecture = typed(counts, CandidateStages.REREAD, gateKinds, roles);
      long attention = architecture.attention();
      PackedWeightLayout.Header header = TestPack.header(architecture);
      assertEquals(
          header.elements(WeightField.TILE_RELATION, 0), (long) counts.tiles() * counts.tiles());
      assertEquals(header.elements(WeightField.CODE_KEY, 0), counts.keyCodes() * attention);
      assertEquals(header.elements(WeightField.CODE_VALUE, 0), counts.keyCodes() * attention);
      assertEquals(
          header.elements(WeightField.REREAD_CODE_KEY, 0), counts.rereadCodes() * attention);
      assertEquals(
          header.elements(WeightField.REREAD_CODE_VALUE, 0), counts.rereadCodes() * attention);
      assertEquals(
          header.elements(WeightField.GATE_KIND_OFFSET, 0), (long) gateKinds * architecture.gate());
      assertEquals(header.elements(WeightField.GATE_OUTPUT_BIAS, 0), gateKinds);
      assertEquals(header.sectionWidth(InputSection.TRANSITION_CODE), counts.fixed());
      header.role(roles - 1, ScoreComposition.TURN_GATE, gateKinds - 1);
      assertThrows(
          IllegalArgumentException.class, () -> header.role(roles, ScoreComposition.ZERO, -1));
      assertThrows(
          IllegalArgumentException.class,
          () -> header.role(0, ScoreComposition.TURN_GATE, gateKinds));
      assertEquals(architecture.copiedEventLayers(), 1);
    }
    assertEquals(THREE_PLAYERS.codeBlocks() * TokenCounts.CODE_BLOCK, 128);
    CollectionArchitecture oneStage = typed(THREE_PLAYERS, CandidateStages.ONE, 7, 9);
    assertThrows(
        IllegalArgumentException.class,
        () -> TestPack.header(oneStage).weight(WeightField.REREAD_OUTPUT, 0, 0));
  }

  /**
   *
   * WeightPack のヘッダーは native の並び（形の記述・入力の列の決まり・領域の幅・手の扱いごとの作り方とゲート・重みの位置・埋め込み）で
   *
   * 書かれ、形の記述から始まる。重みの位置は、重みごとの引き方の項目の数だけ続く。
   *
   */
  @Test
  public void headerEncodesInNativeOrder() {

    CollectionArchitecture architecture = TestPack.architecture(512, TestPack.FOUR_PLAYERS);

    PackedWeightLayout.Header header =
        TestPack.header(architecture)
            .weight(WeightField.QKV, 1, 4096)
            .weight(WeightField.UP, WeightField.ffnEntry(1, TokenGroup.ALL), 8192)
            .weight(WeightField.UP, WeightField.CANDIDATE_FFN, 12288)
            .role(3, ScoreComposition.TURN_GATE, 2)
            .embedding(
                EmbeddingTable.Slot.TILE,
                new EmbeddingTable(
                    InputSection.TILE_CATEGORY,
                    null,
                    null,
                    List.of(
                        new EmbeddingTable.Table(
                            8192, 700, new int[] {0, 1}, new int[] {1, 35}, 24576)),
                    List.of(new EmbeddingTable.Dense(EmbeddingTable.Source.FLAG, 2)),
                    16384,
                    20480,
                    32768));

    ByteBuffer pack = TestPack.direct(PackedWeightLayout.HEADER_BYTES);

    header.write(pack);

    ByteBuffer ints = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);

    int architectureAt = 0;

    int schemaAt = architectureAt + CollectionArchitecture.INTS;

    int sectionsAt = schemaAt + ColumnIndex.values().length;

    int rolesAt = sectionsAt + InputSection.values().length;

    int gatesAt = rolesAt + CollectionArchitecture.MAX_ROLES;

    int weightsAt = gatesAt + CollectionArchitecture.MAX_ROLES;

    int entries = 0;

    for (WeightField weight : WeightField.values()) {

      entries += weight.index().entries();
    }

    int embeddingsAt = weightsAt + entries;

    assertEquals(entries, WeightField.ENTRIES);

    assertEquals(
        embeddingsAt + EmbeddingTable.Slot.values().length * EmbeddingTable.INTS,
        PackedWeightLayout.HEADER_INTS);

    int[] encoded = architecture.encode();

    for (int i = 0; i < encoded.length; i++) {

      assertEquals(ints.getInt((architectureAt + i) * 4), encoded[i], "architecture int " + i);
    }

    assertEquals(ints.getInt((schemaAt + ColumnIndex.RELATION_ROWS.ordinal()) * 4), 112);

    assertEquals(
        ints.getInt((sectionsAt + InputSection.TRANSITION_CODE.ordinal()) * 4),
        architecture.counts().fixed());

    assertEquals(ints.getInt((rolesAt + 3) * 4), ScoreComposition.TURN_GATE.ordinal());

    assertEquals(ints.getInt((gatesAt + 3) * 4), 2);

    assertEquals(ints.getInt((weightsAt + WeightField.QKV.firstEntry() + 1) * 4), 4096);

    assertEquals(ints.getInt((weightsAt + WeightField.QKV.firstEntry()) * 4), -1);

    assertEquals(ints.getInt((weightsAt + WeightField.UP.firstEntry() + 3) * 4), 8192);

    assertEquals(ints.getInt((weightsAt + WeightField.UP.firstEntry() + 9) * 4), 12288);

    int tile = embeddingsAt + EmbeddingTable.Slot.TILE.ordinal() * EmbeddingTable.INTS;

    assertEquals(ints.getInt(tile * 4), InputSection.TILE_CATEGORY.ordinal());

    assertEquals(ints.getInt((tile + 1) * 4), -1);

    assertEquals(ints.getInt((tile + 3) * 4), 1);

    assertEquals(ints.getInt((tile + 4) * 4), 1);

    assertEquals(ints.getInt((tile + 8) * 4), 8192);

    assertEquals(ints.getInt((tile + 9) * 4), 700);

    assertEquals(ints.getInt((tile + 11 + EmbeddingTable.MAX_MEMBERS + 1) * 4), 35);

    assertEquals(ints.getInt((tile + 7) * 4), 32768);

    assertEquals(ints.getInt((tile + 11 + 2 * EmbeddingTable.MAX_MEMBERS) * 4), 24576);

    assertEquals(
        ints.getInt((tile + 8 + EmbeddingTable.MAX_TABLES * EmbeddingTable.TABLE_INTS) * 4), 2);

    assertTrue(
        PackedWeightLayout.HEADER_BYTES >= PackedWeightLayout.HEADER_INTS * 4
            && PackedWeightLayout.HEADER_BYTES % PackedWeightLayout.ALIGNMENT == 0);
  }

  @Test
  public void rejectsDescriptionsTheKernelsCannotRun() {
    EncoderLayerShape all =
        new EncoderLayerShape(LayerRole.ALL_TOKENS, new FeedForwardShape.Shared(512));
    EncoderLayerShape fixed =
        new EncoderLayerShape(LayerRole.FIXED_TOKENS, new FeedForwardShape.Shared(512));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(fixed, all)));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(all, all)));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(all, fixed, fixed, fixed)));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            layers(
                List.of(
                    new EncoderLayerShape(
                        LayerRole.ALL_TOKENS, new FeedForwardShape.Typed(512, 512)),
                    fixed)));
    assertThrows(IllegalArgumentException.class, () -> new FeedForwardShape.Shared(500));
    assertThrows(
        IllegalArgumentException.class,
        () -> typed(new TokenCounts(5, 27, 107, 23, 91), CandidateStages.ONE, 7, 9));
    assertThrows(
        IllegalArgumentException.class,
        () -> typed(new TokenCounts(4, 34, 109, 23, 130), CandidateStages.ONE, 6, 8));
    assertThrows(
        IllegalArgumentException.class,
        () -> typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 6, 17));
    assertThrows(
        IllegalArgumentException.class,
        () -> typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 9, 8));
  }

  /** 範囲の外の数で止めるときの例外のメッセージは、渡した数を示す。 */
  @Test
  public void rejectedCountsReportTheGivenValues() {
    String message =
        expectThrows(IllegalArgumentException.class, () -> new TokenCounts(0, 27, 107, 23, 91))
            .getMessage();
    for (String given :
        new String[] {"seats=0", "tiles=27", "keyCodes=107", "tileRelations=23", "maxEvents=91"}) {
      assertTrue(message.contains(given), message);
    }
  }

  private static CollectionArchitecture layers(List<EncoderLayerShape> layers) {
    return new CollectionArchitecture(
        CollectionArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        layers,
        512,
        CandidateStages.ONE,
        256,
        6,
        8,
        TestPack.FOUR_PLAYERS);
  }
}
