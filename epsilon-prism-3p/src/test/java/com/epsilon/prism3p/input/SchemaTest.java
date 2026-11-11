package com.epsilon.prism3p.input;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism3p.input.InputSchema.ActionRole;
import com.epsilon.prism3p.input.InputSchema.KeyBlock;
import com.epsilon.prism3p.input.InputSchema.OpenSuit;
import com.epsilon.prism3p.input.InputSchema.WaitClass;
import com.epsilon.prism3p.input.InputSchema.WinExact;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの入力の宣言のテスト。列と名前空間に隙間も重なりも無く、語彙は u8 に入る。キーのコードは各ブロックを隙間なく埋め、u8 に入る。
 * 三人打ちの形式が持つ数は、席 3・牌 27・固定のトークン 31・イベントの上限 91 と容量 96・手の扱い 9・ゲート 7・入れ替える色 2 になる。
 * 語彙にチーは無い。
 */
public class SchemaTest {

  private static final Variant VARIANT = Variant.TENHOU_3P;

  @Test
  public void columnsAndNamespacesHaveNoGapsOrOverlaps() {
    assertPartition(InputSchema.ROUND, InputSchema.RoundField.class);
    assertPartition(InputSchema.SEAT, InputSchema.SeatField.class);
    assertPartition(InputSchema.TILE, InputSchema.TileField.class);
    assertPartition(InputSchema.EVENT, InputSchema.EventField.class);
    assertPartition(InputSchema.TRANSITION, InputSchema.TransitionField.class);
    assertPartition(InputSchema.RELATION, InputSchema.RelationAttribute.class);
  }

  @Test
  public void keyCodesFillTheirBlocksWithinOneByte() {
    int key = 0;
    int code = 0;
    for (KeyBlock block : KeyBlock.values()) {
      Assert.assertEquals(block.firstKey(), key, block.name());
      Assert.assertEquals(block.firstCode(), code, block.name());
      key += block.keys();
      code += block.codes();
    }
    Assert.assertEquals(KeyBlock.KEY_COUNT, key);
    Assert.assertEquals(KeyBlock.CODE_COUNT, code);
    Assert.assertTrue(code <= FieldSpec.MAX_VOCABULARY);
    Assert.assertEquals(KeyBlock.roundCode(), KeyBlock.ROUND.firstCode());
    Assert.assertEquals(KeyBlock.seatCode(0, false), KeyBlock.SEAT.firstCode());
    Assert.assertEquals(KeyBlock.seatCode(VARIANT.seats() - 1, true), KeyBlock.SEAT.firstCode() + KeyBlock.SEAT.codes() - 1);
    Assert.assertEquals(KeyBlock.tileCode(0, false, null, false), KeyBlock.TILE.firstCode());
    Assert.assertEquals(
        KeyBlock.tileCode(Tiles.COPIES, true, WaitClass.HAN_6_PLUS, true),
        KeyBlock.TILE.firstCode() + KeyBlock.TILE.codes() - 1);
  }

  @Test
  public void theFormatCarriesTheThreePlayerNumbers() {
    InputFormat format = InputSchema.FORMAT;
    Assert.assertEquals(format.variant(), VARIANT);
    Assert.assertEquals(format.seats(), 3);
    Assert.assertEquals(format.tiles(), 27);
    Assert.assertEquals(format.fixedTokens(), 31);
    Assert.assertEquals(format.keys(), new InputFormat.Keys(1, 3, 27, 107));
    Assert.assertEquals(format.maxEvents(), 91);
    Assert.assertEquals(format.eventCapacity(), 96);
    Assert.assertEquals(format.width(Tensor.TRANSITION_CODE), 31);
    Assert.assertEquals(format.width(Tensor.ACTION_EXACT), Math.max(WinExact.values().length, InputSchema.RiichiExact.values().length));
  }

  /** 牌のトークンは 2〜8 萬を除く 27 種類で、種類の番号の順に並ぶ。入れ替える色は筒子と索子だけ。 */
  @Test
  public void tileTokensLeaveOutTheMiddleManzu() {
    TileTokens tokens = InputSchema.FORMAT.tileTokens();
    Assert.assertEquals(tokens.count(), 27);
    int expected = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (kind >= Tiles.M2 && kind <= Tiles.M8) {
        int absent = kind;
        Assert.expectThrows(IllegalArgumentException.class, () -> tokens.token(absent));
      } else {
        Assert.assertEquals(tokens.token(kind), expected);
        Assert.assertEquals(tokens.kind(expected), kind);
        expected++;
      }
    }
    Assert.assertEquals(tokens.token(Tiles.M1), 0);
    Assert.assertEquals(tokens.token(Tiles.M9), 1);
    Assert.assertEquals(tokens.token(Tiles.P1), 2);
    Assert.assertEquals(tokens.token(Tiles.S1), 11);
    Assert.assertEquals(tokens.token(Tiles.PEI), 23);
    Assert.assertEquals(tokens.movableSuits(), 0b110);
  }

  /** 手の扱いは 9 つで、そのうちゲートは RIICHI・CALL・KAN・TSUMO・RON・KYUSHU・NUKI の 7 つ。チーには手の扱いが無い。 */
  @Test
  public void theActionRolesFollowTheThreePlayerTable() {
    Assert.assertEquals(ActionRole.values().length, 9);
    int gates = 0;
    for (ActionRole role : ActionRole.values()) {
      gates += role.gate() ? 1 : 0;
    }
    Assert.assertEquals(gates, 7);
    Assert.assertTrue(ActionRole.NUKI.gate());
    Assert.assertFalse(ActionRole.NUKI.endsHand());
    Assert.expectThrows(IllegalArgumentException.class, () -> ActionRole.of(Action.Type.CHI));
    Assert.assertEquals(ActionRole.of(Action.Type.NUKI), ActionRole.NUKI);
    for (Action.Type type : Vocabulary.of(Action.Type.class, VARIANT).constants()) {
      Assert.assertNotEquals(type, Action.Type.CHI, "the three-player vocabulary has no chi");
      Assert.assertNotNull(ActionRole.of(type));
    }
  }

  /** イベントの語彙に北抜きがあり、チーが無い。副露の色は、正準の 2 色と萬子と字牌の 4 つ。 */
  @Test
  public void eventAndSuitVocabulariesAreThreePlayer() {
    Vocabulary<PublicEvent> events = Vocabulary.of(PublicEvent.class, VARIANT);
    Assert.assertTrue(events.constants().contains(PublicEvent.NUKI));
    Assert.assertFalse(events.constants().contains(PublicEvent.CHI));
    Assert.assertEquals(OpenSuit.values().length, 4);
  }

  /** どの列も隙間なく 1 回ずつ使われ、名前空間は表の行を隙間なく 1 回ずつ覆う。語彙は u8 に入る。 */
  private static <E extends Enum<E> & BoundedField> void assertPartition(TokenLayout<E> layout, Class<E> type) {
    int[] vocabularies = layout.columnVocabularies();
    boolean[] columns = new boolean[layout.width()];
    boolean[] rows = new boolean[layout.tableSize()];
    for (E field : type.getEnumConstants()) {
      FieldSpec spec = field.spec();
      Assert.assertTrue(spec.vocabulary() <= FieldSpec.MAX_VOCABULARY, field.name());
      for (int index = 0; index < spec.repeats(); index++) {
        int column = layout.column(field, index);
        Assert.assertFalse(columns[column], field.name() + " overlaps column " + column);
        columns[column] = true;
        Assert.assertEquals(vocabularies[column], spec.vocabulary(), field.name());
        int offset = layout.offset(field, index);
        for (int row = offset; row < offset + spec.vocabulary(); row++) {
          Assert.assertFalse(rows[row], field.name() + " overlaps row " + row);
          rows[row] = true;
        }
      }
    }
    for (int column = 0; column < columns.length; column++) {
      Assert.assertTrue(columns[column], type.getSimpleName() + " gap at column " + column);
    }
    for (int row = 0; row < rows.length; row++) {
      Assert.assertTrue(rows[row], type.getSimpleName() + " gap at row " + row);
    }
  }
}
