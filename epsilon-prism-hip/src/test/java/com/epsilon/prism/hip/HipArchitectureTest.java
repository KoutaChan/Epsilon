package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/** 形の記述の決まり、重みの項目、native に渡す int の並び、層の種類で決まるカーネル、数だけで決まる重みと領域の大きさを確かめる。 */
public class HipArchitectureTest {

  private static final HipCounts THREE_PLAYERS = new HipCounts(3, 27, 107, 23, 91);

  private static HipArchitecture typed(HipCounts counts, CandidateStages stages, int gateKinds, int roles) {
    return new HipArchitecture(
        HipArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(
            new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512)),
            new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Typed(1536, 768)),
            new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Typed(1536, 768))),
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
    HipArchitecture v2 = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
    HipArchitecture v3 = typed(TestPack.FOUR_PLAYERS, CandidateStages.REREAD, 6, 8);
    assertTrue(v2.holds(HipWeight.UP, HipWeight.ffnEntry(1, TokenGroup.ALL)));
    assertFalse(v2.holds(HipWeight.UP, HipWeight.ffnEntry(1, TokenGroup.TILE)));
    assertFalse(v2.holds(HipWeight.QKV, 2));
    assertFalse(v2.holds(HipWeight.REREAD_OUTPUT, 0));
    assertTrue(v2.holds(HipWeight.FINAL_NORM, TokenGroup.ALL.ordinal()));
    assertTrue(v3.holds(HipWeight.UP, HipWeight.ffnEntry(2, TokenGroup.TILE)));
    assertFalse(v3.holds(HipWeight.UP, HipWeight.ffnEntry(1, TokenGroup.ALL)));
    assertTrue(v3.holds(HipWeight.UP, HipWeight.CANDIDATE_FFN));
    assertTrue(v3.holds(HipWeight.REREAD_OUTPUT, 0));
    assertFalse(v3.holds(HipWeight.FINAL_NORM, TokenGroup.ALL.ordinal()));
    assertTrue(v3.holds(HipWeight.FINAL_NORM, TokenGroup.CONTEXT.ordinal()));
    assertEquals(v3.ffnWidth(HipWeight.ffnEntry(1, TokenGroup.CONTEXT)), 1536);
    assertEquals(v3.ffnWidth(HipWeight.CANDIDATE_FFN), 1024);
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
    assertEquals(ints.length, HipArchitecture.INTS);
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

  /**
   * カーネルが受け持つ層は層の種類で決まる。全トークンの層は K2・K3、局・席・牌の層は K4、最後でない局・席・牌の層は K3、最後の層は K5 が
   * 受け持つ。
   */
  @Test
  public void kernelsRunTheLayersOfTheirRole() {
    HipArchitecture v2 = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
    HipLayer all = new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512));
    HipArchitecture deep = layers(List.of(all, all, new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(512))));
    HipArchitecture v3 = typed(TestPack.FOUR_PLAYERS, CandidateStages.REREAD, 6, 8);
    HipArchitecture oneStage = typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 6, 8);
    for (HipKernel kernel : List.of(HipKernel.EMBED, HipKernel.RELATION, HipKernel.CANDIDATE, HipKernel.VALUE_GRAPH)) {
      for (HipArchitecture architecture : List.of(v2, deep, v3, oneStage)) {
        assertEquals(accepted(kernel, architecture), List.of(0), kernel + " of " + architecture.layers());
      }
    }
    assertEquals(accepted(HipKernel.ATTENTION_ALL, v2), List.of(0));
    assertEquals(accepted(HipKernel.MLP_ALL, v2), List.of(0));
    assertEquals(accepted(HipKernel.ATTENTION_FIXED, v2), List.of(1));
    assertEquals(accepted(HipKernel.MLP_FIXED, v2), List.of());
    assertEquals(accepted(HipKernel.MLP_LAST, v2), List.of(1));
    assertEquals(accepted(HipKernel.REREAD_KEY_VALUE, v2), List.of());
    assertEquals(accepted(HipKernel.REREAD, v2), List.of());
    assertEquals(accepted(HipKernel.ATTENTION_ALL, deep), List.of(0, 1));
    assertEquals(accepted(HipKernel.MLP_ALL, deep), List.of(0, 1));
    assertEquals(accepted(HipKernel.ATTENTION_FIXED, deep), List.of(2));
    assertEquals(accepted(HipKernel.MLP_FIXED, deep), List.of());
    assertEquals(accepted(HipKernel.MLP_LAST, deep), List.of(2));
    assertEquals(accepted(HipKernel.ATTENTION_ALL, v3), List.of(0));
    assertEquals(accepted(HipKernel.MLP_ALL, v3), List.of(0));
    assertEquals(accepted(HipKernel.ATTENTION_FIXED, v3), List.of(1, 2));
    assertEquals(accepted(HipKernel.MLP_FIXED, v3), List.of(1));
    assertEquals(accepted(HipKernel.MLP_LAST, v3), List.of(2));
    assertEquals(accepted(HipKernel.REREAD_KEY_VALUE, v3), List.of(0));
    assertEquals(accepted(HipKernel.REREAD, v3), List.of(0));
    assertEquals(accepted(HipKernel.REREAD_KEY_VALUE, oneStage), List.of());
    assertEquals(accepted(HipKernel.REREAD, oneStage), List.of());
    assertEquals(deep.allTokenLayers(), 2);
    assertEquals(deep.fixedLayers(), 1);
    assertEquals(deep.fixedIndex(2), 0);
    assertThrows(IllegalArgumentException.class, () -> deep.fixedIndex(1));
  }

  /** カーネルが受け持つ層の番号を、範囲の外の番号も試して返す。 */
  private static List<Integer> accepted(HipKernel kernel, HipArchitecture architecture) {
    List<Integer> layers = new ArrayList<>();
    for (int layer = -1; layer <= HipArchitecture.MAX_DEPTH; layer++) {
      if (kernel.acceptsLayer(layer, architecture)) {
        layers.add(layer);
      }
    }
    return layers;
  }

  /**
   * 重みと領域の大きさは、形の記述の数だけから決まる（四人打ちと、席 3・牌 27・遷移のキーのコード 107・イベント 91・ゲートの種類 7・
   * 手の扱いの種類 9 の三人打ちで確かめる）。牌の関係の表は牌の数の 2 乗、コードの表はキーのコードと 2 段目のコードの数、ゲートの表は
   * ゲートの種類の数、関係のコードは 16 トークンのタイルの数の 2 乗、イベントのキーとバリューは 2 つ目からの局・席・牌の層の数で決まる。
   */
  @Test
  public void sizesFollowTheCounts() {
    for (boolean three : new boolean[] {false, true}) {
      HipCounts counts = three ? THREE_PLAYERS : TestPack.FOUR_PLAYERS;
      int gateKinds = three ? 7 : 6;
      int roles = three ? 9 : 8;
      HipArchitecture architecture = typed(counts, CandidateStages.REREAD, gateKinds, roles);
      long attention = architecture.attention();
      HipPack.Header header = TestPack.header(architecture);
      assertEquals(header.elements(HipWeight.TILE_RELATION, 0), (long) counts.tiles() * counts.tiles());
      assertEquals(header.elements(HipWeight.CODE_KEY, 0), counts.keyCodes() * attention);
      assertEquals(header.elements(HipWeight.CODE_VALUE, 0), counts.keyCodes() * attention);
      assertEquals(header.elements(HipWeight.REREAD_CODE_KEY, 0), counts.rereadCodes() * attention);
      assertEquals(header.elements(HipWeight.REREAD_CODE_VALUE, 0), counts.rereadCodes() * attention);
      assertEquals(header.elements(HipWeight.GATE_KIND_OFFSET, 0), (long) gateKinds * architecture.gate());
      assertEquals(header.elements(HipWeight.GATE_OUTPUT_BIAS, 0), gateKinds);
      assertEquals(header.sectionWidth(HipSection.TRANSITION_CODE), counts.fixed());
      header.role(roles - 1, HipComposition.TURN_GATE, gateKinds - 1);
      assertThrows(IllegalArgumentException.class, () -> header.role(roles, HipComposition.ZERO, -1));
      assertThrows(IllegalArgumentException.class, () -> header.role(0, HipComposition.TURN_GATE, gateKinds));

      int rows = 5;
      int events = 123;
      HipShape shape = new HipShape(architecture, new HipShape.Capacity(rows, events, 40, 40, 4096), 1);
      long fixed = (long) counts.fixed() * rows;
      long tokens = fixed + events;
      long pairs = (long) counts.codeBlocks() * HipCounts.CODE_BLOCK;
      assertEquals(shape.fixedTokens(), counts.fixed());
      assertEquals(HipRegion.RELATION_CODE.capacityBytes(shape), rows * pairs * pairs * Integer.BYTES);
      assertEquals(HipRegion.RESIDUAL.capacityBytes(shape), tokens * architecture.hidden() * 2);
      assertEquals(HipRegion.STATE.capacityBytes(shape), fixed * architecture.hidden() * 2);
      assertEquals(HipRegion.MEMORY.capacityBytes(shape), fixed * 2 * attention * 2);
      assertEquals(architecture.copiedEventLayers(), 1);
      assertEquals(HipRegion.EVENT_KEY_VALUE.capacityBytes(shape), (long) events * 2 * attention * 2);
      assertEquals(HipRegion.REREAD_MEMORY.capacityBytes(shape), tokens * 2 * attention * 2);
    }
    assertEquals(THREE_PLAYERS.codeBlocks() * HipCounts.CODE_BLOCK, 128);
    HipArchitecture oneStage = typed(THREE_PLAYERS, CandidateStages.ONE, 7, 9);
    assertThrows(
        IllegalArgumentException.class, () -> TestPack.header(oneStage).weight(HipWeight.REREAD_OUTPUT, 0, 0));
    HipShape shape = new HipShape(oneStage, new HipShape.Capacity(2, 10, 4, 4, 4096), 1);
    assertEquals(HipRegion.REREAD_MEMORY.capacityBytes(shape), 0L);
  }

  @Test
  public void rejectsDescriptionsTheKernelsCannotRun() {
    HipLayer all = new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512));
    HipLayer fixed = new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(512));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(fixed, all)));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(all, all)));
    assertThrows(IllegalArgumentException.class, () -> layers(List.of(all, fixed, fixed, fixed)));
    assertThrows(
        IllegalArgumentException.class,
        () -> layers(List.of(new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Typed(512, 512)), fixed)));
    assertThrows(IllegalArgumentException.class, () -> new HipFfn.Shared(500));
    assertThrows(IllegalArgumentException.class, () -> typed(new HipCounts(5, 27, 107, 23, 91), CandidateStages.ONE, 7, 9));
    assertThrows(IllegalArgumentException.class, () -> typed(new HipCounts(4, 34, 109, 23, 130), CandidateStages.ONE, 6, 8));
    assertThrows(IllegalArgumentException.class, () -> typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 6, 17));
    assertThrows(IllegalArgumentException.class, () -> typed(TestPack.FOUR_PLAYERS, CandidateStages.ONE, 9, 8));
  }

  /** 範囲の外の数で止めるときの例外のメッセージは、渡した数を示す。 */
  @Test
  public void rejectedCountsReportTheGivenValues() {
    String message = expectThrows(IllegalArgumentException.class, () -> new HipCounts(0, 27, 107, 23, 91)).getMessage();
    for (String given : new String[] {"seats=0", "tiles=27", "keyCodes=107", "tileRelations=23", "maxEvents=91"}) {
      assertTrue(message.contains(given), message);
    }
  }

  private static HipArchitecture layers(List<HipLayer> layers) {
    return new HipArchitecture(
        HipArchitecture.HIDDEN, 1e-5f, 0.25f, 0.01f, layers, 512, CandidateStages.ONE, 256, 6, 8, TestPack.FOUR_PLAYERS);
  }
}
