package com.epsilon.prism.backend;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static com.epsilon.prism.input.InputSchema.THREE_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.ColumnIndex;
import com.epsilon.prism.kernel.InputSection;
import com.epsilon.prism.kernel.PackedWeightLayout;
import com.epsilon.prism.kernel.ScoreComposition;
import com.epsilon.prism.kernel.TokenCounts;
import com.epsilon.prism.kernel.WeightField;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.util.EnumMap;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 三人打ちのネットワークの WeightLayout と WeightPack を確かめる（CPU、native なし）。v2 の形と V3 で、推論のパラメーターが全て置かれ、
 * ヘッダーに書く列の番号・領域の幅・手の扱いごとの作り方とゲートが三人打ちの入力形式と合う。牌の関係の表は 27² で {@link TileTokens} と
 * 同じで、手の扱い・ゲート・安全度の数がアーキテクチャの記述と合う。形式と数が合わない記述は、英語の例外で拒む。
 */
public class ThreePlayerWeightLayoutTest {

  private static final InputFormat FORMAT = THREE_PLAYER;

  /** 三人打ちの手の扱いごとの、ゲートの種類の番号。ゲートになる扱いを宣言の順に数えた番号で、ゲートでなければ −1。作り方は次の表に持つ。 */
  private static final Map<ActionRole, Integer> GATE_KINDS = new EnumMap<>(ActionRole.class);

  private static final Map<ActionRole, ScoreComposition> COMPOSITIONS =
      new EnumMap<>(ActionRole.class);

  static {
    GATE_KINDS.put(ActionRole.DISCARD, -1);
    GATE_KINDS.put(ActionRole.RIICHI, 0);
    GATE_KINDS.put(ActionRole.CALL, 1);
    GATE_KINDS.put(ActionRole.KAN, 2);
    GATE_KINDS.put(ActionRole.TSUMO, 3);
    GATE_KINDS.put(ActionRole.RON, 4);
    GATE_KINDS.put(ActionRole.KYUSHU, 5);
    GATE_KINDS.put(ActionRole.NUKI, 6);
    GATE_KINDS.put(ActionRole.PASS, -1);
    COMPOSITIONS.put(ActionRole.DISCARD, ScoreComposition.DISCARD_VALUE);
    COMPOSITIONS.put(ActionRole.RIICHI, ScoreComposition.RIICHI_GATE);
    COMPOSITIONS.put(ActionRole.CALL, ScoreComposition.RESPONSE_GATE);
    COMPOSITIONS.put(ActionRole.KAN, ScoreComposition.TURN_GATE);
    COMPOSITIONS.put(ActionRole.TSUMO, ScoreComposition.TURN_GATE);
    COMPOSITIONS.put(ActionRole.RON, ScoreComposition.RESPONSE_GATE);
    COMPOSITIONS.put(ActionRole.KYUSHU, ScoreComposition.TURN_GATE);
    COMPOSITIONS.put(ActionRole.NUKI, ScoreComposition.TURN_GATE);
    COMPOSITIONS.put(ActionRole.PASS, ScoreComposition.ZERO);
  }

  @DataProvider
  public Object[][] arms() {
    return new Object[][] {{ThreePlayerRows.Arm.V2}, {ThreePlayerRows.Arm.V3}};
  }

  @Test(dataProvider = "arms")
  public void everyParameterIsPlacedAndTheHeaderFollowsTheThreePlayerFormat(
      ThreePlayerRows.Arm arm) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, arm.shape(), 11 + arm.ordinal());
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 7);
      Assert.assertSame(layout.format(), FORMAT);
      PackReadBack.assertReadsBack(layout, pack, network.inferenceParameters());

      CollectionArchitecture architecture = layout.architecture();
      TokenCounts counts = architecture.counts();
      Assert.assertEquals(counts.seats(), 3);
      Assert.assertEquals(counts.tiles(), 27);
      Assert.assertEquals(counts.fixed(), 31);
      Assert.assertEquals(counts.tileRelations(), MahjongTileRelation.values().length);
      Assert.assertEquals(counts.maxEvents(), FORMAT.maxEvents());
      Assert.assertEquals(counts, PrismShape.counts(FORMAT));
      Assert.assertEquals(counts.opponents(), 2);
      Assert.assertEquals(
          FORMAT.relation().spec(RelationAttribute.SAFETY).repeats(), counts.opponents());
      Assert.assertEquals(architecture.roles(), THREE_PLAYER.roles().size());
      Assert.assertEquals(architecture.gateKinds(), 7);

      PackView view = new PackView(pack.buffer());
      Assert.assertEquals(view.architecture(), architecture.encode());
      PackedWeightLayout.Header header = layout.header();
      for (ColumnIndex field : ColumnIndex.values()) {
        Assert.assertEquals(view.schema(field), header.schema(field), field.name());
      }
      Assert.assertEquals(
          view.schema(ColumnIndex.ROW_FIRST_EVENT), RowColumn.FIRST_EVENT.ordinal());
      Assert.assertEquals(view.schema(ColumnIndex.ROW_ACTIONS), RowColumn.ACTIONS.ordinal());
      Assert.assertEquals(view.schema(ColumnIndex.ACTION_ROLE), ActionColumn.ROLE.ordinal());
      Assert.assertEquals(
          view.schema(ColumnIndex.ACTION_REPRESENTATIVE), ActionColumn.REPRESENTATIVE.ordinal());
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_TILE),
          FORMAT.relation().column(RelationAttribute.TILE, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_OWNER),
          FORMAT.relation().column(RelationAttribute.OWNER, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_EVENT),
          FORMAT.relation().column(RelationAttribute.EVENT, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_RIICHI),
          FORMAT.relation().column(RelationAttribute.OWNER_RIICHI_EVENT, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_LAST_DISCARD),
          FORMAT.relation().column(RelationAttribute.OWNER_LAST_DISCARD_EVENT, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_LINK),
          FORMAT.relation().column(RelationAttribute.LINK, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.RELATION_SAFETY),
          FORMAT.relation().column(RelationAttribute.SAFETY, 0));
      Assert.assertEquals(
          view.schema(ColumnIndex.TRANSITION_CALL_DISCARD),
          FORMAT.transition().column(TransitionField.CALL_DISCARD));
      Assert.assertEquals(
          view.schema(ColumnIndex.TRANSITION_MAIN_TILE),
          FORMAT.transition().column(TransitionField.MAIN_TILE));
      for (Tensor tensor : Tensor.values()) {
        InputSection section = InputSection.valueOf(tensor.name());
        int expected =
            section == InputSection.TRANSITION_CODE ? counts.fixed() : FORMAT.width(tensor);
        Assert.assertEquals(view.width(section), expected, section.name());
      }
      Assert.assertEquals(FORMAT.roles(), THREE_PLAYER.roles());
      for (ActionRole role : THREE_PLAYER.roles()) {
        Assert.assertEquals(
            view.composition(THREE_PLAYER.roleIndex(role)), COMPOSITIONS.get(role), role.name());
        Assert.assertEquals(
            view.gate(THREE_PLAYER.roleIndex(role)), (int) GATE_KINDS.get(role), role.name());
        Assert.assertEquals(
            WeightLayout.gateKind(FORMAT, role), (int) GATE_KINDS.get(role), role.name());
      }
    }
  }

  /**
   * 牌の関係の表は 27² の大きさで、{@link TileTokens} の三人打ちの表と同じ。中身は、四人打ちの 34² の表
   * （{@link MahjongTileRelation#relationIds()}）から三人打ちの牌の種類の行と列を抜き出したものとも一致する。
   */
  @Test(dataProvider = "arms")
  public void theTileRelationTableIsTheThreePlayerTokens(ThreePlayerRows.Arm arm) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = ThreePlayerRows.network(manager, arm.shape(), 13 + arm.ordinal());
      WeightLayout layout = WeightLayout.of(network);
      PackView view = new PackView(WeightPack.build(layout, network, 3).buffer());
      TileTokens tokens = TileTokens.of(TENHOU_3P);
      int[] relations = tokens.relations();
      Assert.assertEquals(tokens.count(), 27);
      Assert.assertEquals(relations.length, 27 * 27);
      Assert.assertEquals(layout.header().elements(WeightField.TILE_RELATION, 0), 27L * 27);
      long[] four = MahjongTileRelation.relationIds();
      for (int query = 0; query < 27; query++) {
        for (int key = 0; key < 27; key++) {
          int i = query * 27 + key;
          Assert.assertEquals(
              view.u8(WeightField.TILE_RELATION, 0, i), relations[i], "tile relation " + i);
          Assert.assertEquals(
              relations[i],
              four[tokens.kind(query) * Tiles.KINDS + tokens.kind(key)],
              "tile relation " + i);
        }
      }
    }
  }

  /** 手の扱いの数・ゲートの種類の数・イベントの上限・安全度の列がアーキテクチャの記述と合わない形式は、英語の例外で拒む。 */
  @Test
  public void aFormatThatDoesNotMatchTheArchitectureIsRefused() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(FORMAT, PrismShape.standard(2));
      network.initialize(manager);
      CollectionArchitecture three = WeightLayout.of(network).architecture();
      TokenCounts counts = three.counts();

      assertRefused(
          FORMAT,
          with(three, three.gateKinds(), three.roles() - 1, counts),
          network,
          "the TENHOU_3P input declares 9 action roles with 7 gate kinds, the architecture has 8"
              + " with 7");
      assertRefused(
          FORMAT,
          with(three, three.gateKinds() - 1, three.roles(), counts),
          network,
          "the TENHOU_3P input declares 9 action roles with 7 gate kinds, the architecture has 9"
              + " with 6");
      TokenCounts longer =
          new TokenCounts(
              counts.seats(),
              counts.tiles(),
              counts.keyCodes(),
              counts.tileRelations(),
              counts.maxEvents() + 1);
      assertRefused(
          FORMAT,
          with(three, three.gateKinds(), three.roles(), longer),
          network,
          "the TENHOU_3P input has the counts " + counts + ", the architecture has " + longer);
    }
  }

  private static void assertRefused(
      InputFormat format,
      CollectionArchitecture architecture,
      PrismNetwork network,
      String message) {
    IllegalArgumentException refused =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> WeightLayout.of(format, architecture, network.inferenceParameters()));
    Assert.assertEquals(refused.getMessage(), message);
  }

  private static CollectionArchitecture with(
      CollectionArchitecture a, int gateKinds, int roles, TokenCounts counts) {
    return new CollectionArchitecture(
        a.hidden(),
        a.epsilon(),
        a.scale(),
        a.tau(),
        a.layers(),
        a.candidateFfn(),
        a.stages(),
        a.gate(),
        gateKinds,
        roles,
        counts);
  }
}
