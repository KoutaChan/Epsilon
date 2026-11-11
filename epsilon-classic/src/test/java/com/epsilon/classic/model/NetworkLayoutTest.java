package com.epsilon.classic.model;

import static org.testng.Assert.assertEquals;

import ai.djl.nn.Block;
import com.epsilon.classic.Architecture;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 子の層の名前と登録の順、分岐とノードの並びが変わらないことを確かめる。変換した重みは名前で読み、DJL は登録の順の番号を名前の先頭に
 * 付けるので、順が変わると読めなくなる。ネットワークを組むだけで、ネイティブのライブラリは使わない。
 */
public class NetworkLayoutTest {

  private static final List<String> NANO_NETWORK =
      List.of(
          "01stateEncoder",
          "02policyStateReadout",
          "03valueStateReadout",
          "04policyHead",
          "05valueHiddenHead",
          "06valueHead");

  private static final List<String> MAJOR_NETWORK =
      List.of(
          "01stateEncoder",
          "02policyStateReadout",
          "03valueStateReadout",
          "04policyHead",
          "05valueHiddenHead",
          "06valueResidualHead",
          "07valueHead");

  private static final List<String> NANO_STATE =
      List.of(
          "01stateFeatureEmbedding",
          "02canonicalTileEmbedding",
          "03roundFeatureProjection",
          "04playerFeatureProjection",
          "05tileFeatureProjection",
          "06tileRelationEncoder",
          "07riverEventProjection",
          "08meldFeatureProjection",
          "09entityTokenLayerNorm",
          "10playerMemoryEncoder",
          "11strategicContextEncoder");

  private static final List<String> MAJOR_STATE =
      List.of(
          "01stateFeatureEmbedding",
          "02canonicalTileEmbedding",
          "03roundFeatureProjection",
          "04playerFeatureProjection",
          "05tileFeatureProjection",
          "06tileRelationEncoder",
          "07riverEventProjection",
          "08meldFeatureProjection",
          "09entityTokenLayerNorm",
          "10playerMemoryEncoder",
          "11playerMemoryKeyValueProjection",
          "12publicHistoryEncoder",
          "13strategicContextEncoder");

  private static final List<String> NANO_POLICY =
      List.of(
          "01actionFeatureEmbedding",
          "02transitionTileRelationEmbedding",
          "03transitionWaitYakuProjection",
          "04policyNodeEmbedding",
          "05alternativeEmbedding",
          "06actionFeatureProjection",
          "07transitionFeatureProjection",
          "08transitionTileKeyValueProjection",
          "09transitionTileQueryProjection",
          "10transitionWaitKeyValueProjection",
          "11playerTileQueryProjection",
          "12playerMemoryKeyValueProjection",
          "13playerTileContextMixer",
          "14transitionFusion",
          "15transitionLayerNorm",
          "16discardCandidateScorer",
          "17candidateFusion",
          "18candidateLayerNorm",
          "19actionCandidateHidden",
          "20actionCandidateScoreHead",
          "21alternativeHidden",
          "22alternativeScoreHead",
          "23ronGate",
          "24callGate",
          "25tsumoGate",
          "26kyushuGate",
          "27kanGate",
          "28riichiGate");

  private static final List<String> MAJOR_POLICY =
      List.of(
          "01actionFeatureEmbedding",
          "02transitionTileRelationEmbedding",
          "03pointProjection",
          "04pointOutcomeEncoder",
          "05alternativeOffset",
          "06actionFeatureProjection",
          "07transitionFeatureProjection",
          "08transitionTileKeyValueProjection",
          "09transitionPublicContextProjection",
          "10transitionTileQueryProjection",
          "11transitionWaitKeyValueProjection",
          "12playerTileQueryProjection",
          "13playerTileContextMixer",
          "14transitionFusion",
          "15transitionLayerNorm",
          "16discardCandidateScorer",
          "17candidateFusion",
          "18candidateLayerNorm",
          "19actionCandidateHidden",
          "20actionCandidateScoreHead",
          "21alternativeHidden",
          "22alternativeScoreHead",
          "23ronGate",
          "24callGate",
          "25tsumoGate",
          "26kyushuGate",
          "27kanGate",
          "28riichiGate");

  @Test
  public void childrenKeepTheConvertedOrder() {
    ClassicNetwork nano = new ClassicNetwork(Architecture.NANO, Variant.TENHOU_4P, RankUtility.TENHOU);
    assertEquals(children(nano), NANO_NETWORK);
    assertEquals(children(child(nano, "01stateEncoder")), NANO_STATE);
    assertEquals(children(child(nano, "04policyHead")), NANO_POLICY);
    ClassicNetwork major = new ClassicNetwork(Architecture.MAJOR, Variant.TENHOU_4P, RankUtility.AVERAGE_RANK);
    assertEquals(children(major), MAJOR_NETWORK);
    assertEquals(children(child(major, "01stateEncoder")), MAJOR_STATE);
    assertEquals(children(child(major, "04policyHead")), MAJOR_POLICY);
  }

  @Test
  public void gatesKeepTheirParts() {
    ClassicNetwork major = new ClassicNetwork(Architecture.MAJOR, Variant.TENHOU_4P, RankUtility.AVERAGE_RANK);
    Block policy = child(major, "04policyHead");
    assertEquals(children(child(policy, "23ronGate")), List.of("01hidden", "02directFeatureProjection", "03score"));
    assertEquals(children(child(policy, "24callGate")), List.of("01hidden", "02score"));
    ClassicNetwork nano = new ClassicNetwork(Architecture.NANO, Variant.TENHOU_4P, RankUtility.TENHOU);
    assertEquals(children(child(child(nano, "04policyHead"), "23ronGate")), List.of("01hidden", "02score"));
    assertEquals(
        children(child(child(nano, "01stateEncoder"), "11strategicContextEncoder")),
        List.of(
            "01handQueryProjection",
            "02handKeyValueProjection",
            "03handContextProjection",
            "04handSummaryLayerNorm",
            "05tokenRoleEmbedding",
            "06block0",
            "07block1"));
    assertEquals(children(child(child(major, "01stateEncoder"), "13strategicContextEncoder")).size(), 8);
  }

  @Test
  public void alternativesAndNodesKeepTheirOrder() {
    assertEquals(
        Arrays.toString(Alternative.values()),
        "[RON, TSUMO, KYUSHU, CALL, CHI, PON, DAIMINKAN, KAN, ANKAN, KAKAN]");
    assertEquals(
        Arrays.toString(PolicyNode.values()),
        "[RON_GATE, TSUMO_GATE, KYUSHU_GATE, CALL_GATE, MELD_TYPE, MELD_CANDIDATE, KAN_GATE, KAN_TYPE,"
            + " DISCARD_IDENTITY, RIICHI_GATE]");
  }

  private static List<String> children(Block block) {
    return block.getChildren().keys();
  }

  private static Block child(Block block, String name) {
    return block.getChildren().get(name);
  }
}
