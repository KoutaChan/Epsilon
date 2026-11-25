package com.epsilon.prism.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static com.epsilon.prism.input.InputSchema.THREE_PLAYER;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Condition;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Reduction;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.OpenSuit;
import com.epsilon.prism.input.InputSchema.WaitClass;
import com.epsilon.prism.input.TileKeyCode.Digit;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 入力、補助のヘッド、記録 O のスキーマのテスト。列と名前空間に隙間も重なりも無く、語彙が u8 に収まる。キーのコードが各ブロックを隙間なく
 * 埋めて u8 に収まる。候補のヘッドが隙間なく 58 列を覆う。役の列がエンジンの役の順と同じ。条件ごとに当てはまる結果の数が、手で数えた数と合う。
 */
public class SchemaTest {

  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void encodedDecisionHistoryMatchesReference(Variant variant) throws Exception {
    var base = InputFormat.of(variant);
    if (variant == Variant.TENHOU_3P) {
      var engine = PlayedScene.nukiResponse().engine;
      InputSamples.assertHistory(
          base,
          base.description(),
          new RowEncoder(base).capture(engine, engine.decisions().slotOf(1)));
    } else {
      InputSamples.assertHistory(base, base.description());
    }
  }

  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void preservesBaseBytes(Variant variant) throws Exception {
    InputSamples.assertBaseBytes(
        InputFormat.of(variant),
        variant == Variant.TENHOU_3P
            ? "c20bfc89177f020211a630d2af10254ffd63a119fda161395077457a8b3ecd8c"
            : "b07d232f9e7fc5734b8dd22f1f97f689c03026d456fa831d7c3aa36a293f3002");
  }

  @Test(dataProvider = "formats", dataProviderClass = InputFormatTest.class)
  public void columnsAndNamespacesHaveNoGapsOrOverlaps(InputFormat format) {
    assertPartition(format.round(), InputSchema.RoundField.class);
    assertPartition(format.seat(), InputSchema.SeatField.class);
    assertPartition(format.tile(), InputSchema.TileField.class);
    assertPartition(format.event(), InputSchema.EventField.class);
    assertPartition(format.transition(), InputSchema.TransitionField.class);
    assertPartition(format.relation(), InputSchema.RelationAttribute.class);
  }

  @Test(dataProvider = "formats", dataProviderClass = InputFormatTest.class)
  public void keyCodesFillTheirBlocksWithinOneByte(InputFormat format) {
    int key = 0;
    int code = 0;
    for (KeyBlock keyBlock : KeyBlock.values()) {
      var block = format.block(keyBlock);
      Assert.assertEquals(block.firstKey(), key, keyBlock.name());
      Assert.assertEquals(block.firstCode(), code, keyBlock.name());
      key += block.keys();
      code += block.codes();
    }
    Assert.assertEquals(format.keys().count(), key);
    Assert.assertEquals(format.keys().codes(), code);
    Assert.assertTrue(code <= FieldSpec.MAX_VOCABULARY);
    Assert.assertEquals(format.seatCode(0, false), format.block(KeyBlock.SEAT).firstCode());
    Assert.assertEquals(
        format.seatCode(format.seats() - 1, true),
        format.block(KeyBlock.SEAT).firstCode() + format.block(KeyBlock.SEAT).codes() - 1);
    Assert.assertEquals(
        format.tileCode(0, false, null, false), format.block(KeyBlock.TILE).firstCode());
    Assert.assertEquals(
        format.tileCode(Tiles.COPIES, true, WaitClass.HAN_6_PLUS, true),
        format.block(KeyBlock.TILE).firstCode() + format.block(KeyBlock.TILE).codes() - 1);
  }

  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void segmentedKeysMatchPublicTileAndSeatFactsAcrossRows(Variant variant) {
    var baseFormat = InputFormat.of(variant);
    var format = InputFormat.of(variant, Set.of(InputFeature.CANDIDATE_SEGMENTS));
    var batch =
        InputSamples.encode(
            HostInputs.allocate(format, InputSamples.CAPACITY, ByteBuffer::allocate));
    var base =
        InputSamples.encode(
            HostInputs.allocate(baseFormat, InputSamples.CAPACITY, ByteBuffer::allocate));
    var plan = batch.plan();
    var blob = batch.blob();
    int keys = format.fixedTokens();
    int tiles = format.tiles();
    int keyWidth = format.width(Tensor.TRANSITION_CODE);
    int ownRiver = baseFormat.tile().column(InputSchema.TileField.OWN_RIVER);
    int dora = baseFormat.tile().column(InputSchema.TileField.DORA);
    int riichi = baseFormat.seat().column(InputSchema.SeatField.RIICHI);
    int doraRadix = baseFormat.tile().spec(InputSchema.TileField.DORA).vocabulary();
    int north = variant == TENHOU_3P ? format.tileTokens().token(Tiles.PEI) : -1;
    int usedUkeire = 0;
    for (int row = 0; row < plan.rows(); row++) {
      for (int tr = plan.firstTransition(row);
          tr < plan.firstTransition(row) + plan.transitions(row);
          tr++) {
        int start = plan.offset(Tensor.TRANSITION_CODE) + tr * keyWidth;
        Assert.assertEquals(
            blob.slice(start, keys),
            base.blob().slice(base.plan().offset(Tensor.TRANSITION_CODE) + tr * keys, keys));
        for (int t = 0; t < tiles; t++) {
          int code =
              Byte.toUnsignedInt(blob.get(start + keys - tiles + t))
                  - baseFormat.block(KeyBlock.TILE).firstCode();
          var space = format.tileKeyCode().space();
          int ukeire = space.digit(code, Digit.UKEIRE);
          usedUkeire += ukeire;
          int tile =
              plan.offset(Tensor.TILE_CATEGORY) + (row * tiles + t) * baseFormat.tile().width();
          Assert.assertEquals(blob.get(start + keys + t), ukeire * 2 + blob.get(tile + ownRiver));
          Assert.assertEquals(
              blob.get(start + keys + tiles + t),
              ukeire * doraRadix + blob.get(tile + dora) - (t == north ? 1 : 0));
          for (int s = 0; s < format.seats(); s++) {
            int seat =
                plan.offset(Tensor.SEAT_CATEGORY)
                    + (row * format.seats() + s) * baseFormat.seat().width();
            int ready = blob.get(seat + riichi) > 1 ? 1 : 0;
            int seg = Byte.toUnsignedInt(blob.get(start + keys + (2 + s) * tiles + t));
            Assert.assertEquals(seg & 1, ready);
            Assert.assertEquals(
                seg >> 3,
                (space.digit(code, Digit.WAIT_CLASS) > 0 ? 2 : 0)
                    + space.digit(code, Digit.DISCARDED));
            if (s > 0) {
              Assert.assertEquals(
                  (seg >> 1) & 3,
                  blob.get(tile + baseFormat.tile().column(InputSchema.TileField.SAFETY, s - 1)));
            } else {
              Assert.assertEquals(((seg >> 1) & 3) == 1, blob.get(tile + ownRiver) != 0);
            }
          }
        }
      }
    }
    Assert.assertTrue(usedUkeire > 0);
  }

  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void causalPublicHistoryOnlyAppendsAcrossDecisions(Variant variant) {
    var baseFormat = InputFormat.of(variant);
    var format = InputFormat.of(variant, Set.of(InputFeature.PUBLIC_ORDER_EVENTS));
    var host = HostInputs.allocate(format, RowEncoder.ONE_ROW, ByteBuffer::allocate);
    var writer = format.writer();
    var capture = new RowEncoder(baseFormat);
    int[] callsAndIndicators = new int[2];
    new TableWalk(variant, TableWalk.Policy.CALLS)
        .run(
            9101,
            1,
            new TableWalk.Visitor() {
              final byte[][] previous = new byte[format.seats()][];

              @Override
              public void roundEnd(Engine engine) {
                Arrays.fill(previous, null);
              }

              @Override
              public void decision(Engine engine, int slot) {
                Observation o = capture.capture(engine, slot);
                var plan = host.plan();
                plan.clear();
                Assert.assertEquals(plan.add(o), 0);
                plan.seal();
                writer.write(o, host, 0);
                var blob = host.input().blob();
                int cats = format.event().width();
                int attrs = format.relation().width();
                int stride = cats + Short.BYTES + attrs;
                byte[] history = new byte[o.publicEvents() * stride];
                for (int e = 0; e < o.publicEvents(); e++) {
                  int cat = plan.offset(Tensor.EVENT_CATEGORY) + e * cats;
                  int at = e * stride;
                  blob.get(cat, history, at, cats);
                  blob.get(
                      plan.offset(Tensor.EVENT_SCALAR) + e * Short.BYTES,
                      history,
                      at + cats,
                      Short.BYTES);
                  blob.get(
                      plan.offset(Tensor.RELATION) + (plan.tokenStart(Axis.EVENT) + e) * attrs,
                      history,
                      at + cats + Short.BYTES,
                      attrs);
                  if (blob.get(cat + baseFormat.event().column(InputSchema.EventField.CALLED_BY))
                      != 0) {
                    int kind =
                        blob.get(cat + baseFormat.event().column(InputSchema.EventField.KIND));
                    var kindValues = format.event().spec(InputSchema.EventField.KIND).values();
                    Enum<?> event = kindValues.get(kind - 1);
                    Assert.assertTrue(
                        event == PublicEvent.CHI
                            || event == PublicEvent.PON
                            || event == PublicEvent.DAIMINKAN,
                        "caller belongs to an open meld");
                    callsAndIndicators[0]++;
                  }
                }
                callsAndIndicators[1] += o.kanDoraIndicators();
                int seat = o.observer();
                if (previous[seat] != null) {
                  Assert.assertTrue(history.length >= previous[seat].length);
                  Assert.assertEquals(
                      Arrays.mismatch(
                          previous[seat],
                          0,
                          previous[seat].length,
                          history,
                          0,
                          previous[seat].length),
                      -1,
                      "public event prefix for seat " + seat);
                }
                previous[seat] = history;
              }
            });
    Assert.assertTrue(callsAndIndicators[0] > 0 && callsAndIndicators[1] > 0);
  }

  @Test
  public void candidateHeadsSpanFiftyEightColumns() {
    TargetSchema heads = TargetSchema.of(Variant.TENHOU_4P);
    int offset = 0;
    for (CandidateTarget target : CandidateTarget.values()) {
      Assert.assertEquals(heads.offset(target), offset, target.name());
      Assert.assertEquals(heads.width(target), target.width(Variant.TENHOU_4P), target.name());
      offset += heads.width(target);
    }
    Assert.assertEquals(offset, 58);
    Assert.assertEquals(heads.candidateWidth(), 58);
    Assert.assertEquals(heads.width(CandidateTarget.WINNER), 3);
    int reading = 0;
    for (ReadingTarget target : ReadingTarget.values()) {
      Assert.assertEquals(heads.offset(target), reading, target.name());
      reading += heads.width(target);
    }
    Assert.assertEquals(heads.readingWidth(), reading);
    Assert.assertEquals(heads.opponents(), 3);
  }

  /** 役はまとめない。役のヘッドの列はエンジンの役 42 種で、列 k は役のビットの k 番目。損失は列ごとの損失の和にする。 */
  @Test
  public void yakuColumnsFollowTheEngine() {
    Assert.assertEquals(CandidateTarget.YAKU.width(Variant.TENHOU_4P), 42);
    Assert.assertEquals(CandidateTarget.YAKU.width(Variant.TENHOU_4P), ScoringYaku.values().length);
    for (ScoringYaku yaku : ScoringYaku.values()) {
      Assert.assertEquals(yaku.bit(), 1L << yaku.ordinal(), yaku.name());
    }
    Assert.assertEquals(CandidateTarget.YAKU.reduction(), Reduction.SUM);
    Assert.assertEquals(ReadingTarget.DANGER.reduction(), Reduction.MEAN);
  }

  @Test
  public void conditionsSelectOutcomes() {
    Assert.assertEquals(outcomes(Condition.ALWAYS), Outcome.values().length);
    Assert.assertEquals(outcomes(Condition.SELF_WIN), 2);
    Assert.assertEquals(outcomes(Condition.DEAL_IN), 1);
    Assert.assertEquals(outcomes(Condition.OTHER_WIN), 3);
    Assert.assertTrue(Condition.OTHER_WIN.holds(Outcome.DEAL_IN));
  }

  @Test
  public void theFormatCarriesTheThreePlayerNumbers() {
    InputFormat format = THREE_PLAYER;
    Assert.assertEquals(format.variant(), TENHOU_3P);
    Assert.assertEquals(format.seats(), 3);
    Assert.assertEquals(format.tiles(), 27);
    Assert.assertEquals(format.fixedTokens(), 31);
    Assert.assertEquals(format.keys(), new InputFormat.Keys(1, 3, 27, 107));
    Assert.assertEquals(format.maxEvents(), 91);
    Assert.assertEquals(format.eventCapacity(), 96);
    Assert.assertEquals(format.width(Tensor.TRANSITION_CODE), 31);
    Assert.assertEquals(
        format.width(Tensor.ACTION_EXACT),
        Math.max(THREE_PLAYER.winFields().size(), InputSchema.RiichiExact.values().length));
  }

  /** 牌のトークンは 2〜8 萬を除く 27 種類で、種類の番号の順に並ぶ。入れ替える色は筒子と索子だけ。 */
  @Test
  public void tileTokensLeaveOutTheMiddleManzu() {
    TileTokens tokens = THREE_PLAYER.tileTokens();
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
    Assert.assertEquals(THREE_PLAYER.roles().size(), 9);
    int gates = 0;
    for (ActionRole role : THREE_PLAYER.roles()) {
      gates += role.gate() ? 1 : 0;
    }
    Assert.assertEquals(gates, 7);
    Assert.assertTrue(ActionRole.NUKI.gate());
    Assert.assertFalse(ActionRole.NUKI.endsHand());
    Assert.expectThrows(IllegalArgumentException.class, () -> THREE_PLAYER.role(Action.Type.CHI));
    Assert.assertEquals(THREE_PLAYER.role(Action.Type.NUKI), ActionRole.NUKI);
    for (Action.Type type : Vocabulary.of(Action.Type.class, TENHOU_3P).constants()) {
      Assert.assertNotEquals(type, Action.Type.CHI, "the three-player vocabulary has no chi");
      Assert.assertNotNull(THREE_PLAYER.role(type));
    }
  }

  /** イベントの語彙に北抜きがあり、チーが無い。副露の色は、正準の 2 色と萬子と字牌の 4 つ。 */
  @Test
  public void eventAndSuitVocabulariesAreThreePlayer() {
    Vocabulary<PublicEvent> events = Vocabulary.of(PublicEvent.class, TENHOU_3P);
    Assert.assertTrue(events.constants().contains(PublicEvent.NUKI));
    Assert.assertFalse(events.constants().contains(PublicEvent.CHI));
    Assert.assertEquals(Vocabulary.of(OpenSuit.class, TENHOU_3P).size(), 4);
  }

  /** 列が隙間なく 1 回ずつ使われ、名前空間が表の行を隙間なく 1 回ずつ覆うことを確かめる。語彙は u8 に収まる。 */
  private static <E extends Enum<E> & BoundedField> void assertPartition(
      TokenLayout<E> layout, Class<E> type) {
    int[] vocabularies = layout.columnVocabularies();
    boolean[] columns = new boolean[layout.width()];
    boolean[] rows = new boolean[layout.tableSize()];
    for (E field : layout.fields()) {
      FieldSpec spec = layout.spec(field);
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

  private static int outcomes(Condition condition) {
    int count = 0;
    for (Outcome outcome : Outcome.values()) {
      if (condition.holds(outcome)) {
        count++;
      }
    }
    return count;
  }
}
