package com.epsilon.agent.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.ObservationSchema.EventBits;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.ObservationSchema.HeaderBits;
import com.epsilon.agent.observe.ObservationSchema.LastEvent;
import com.epsilon.agent.observe.ObservationSchema.LegalBits;
import com.epsilon.agent.observe.ObservationSchema.WinBits;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Optional;
import org.testng.annotations.Test;

/**
 * 観測のスキーマのテスト。四人打ちのフィンガープリントは、続いている系譜のコホートやスナップショットを読む側との契約なので、
 * 文字列の定数で固定する（PRISM が書く値と同じ）。記述の文字列も固定するので、変わったときは、どのフィールドが変わったかがテストの
 * 差分に出る。
 * 三人打ちの幅は {@code Variant} から取る。
 */
public class ObservationSchemaTest {

  private static final String FOUR_PLAYER_FINGERPRINT = "bb3db98cb8b5b43a";

  /** 四人打ちの観測のスキーマの記述。フィンガープリントはこの文字列から作る。 */
  private static final String FOUR_PLAYER_DESCRIPTION =
      "epsilon-prism-v2-observation"
          + ";ObservationSchema$Section=[HEADER, SEAT, HAND, EVENT, LEGAL, CALL_DISCARD, WIN]"
          + ";ObservationSchema$HeaderBits{unit=64,units=2,DECISION_KIND:0..3@0,OBSERVER:0..3@2,DEALER:0..3@4,"
          + "KYOKU:0..11@6,HONBA:0..255@10,KYOTAKU:0..100@18,WALL:0..70@25,FURITEN_TEMPORARY:0..1@32,"
          + "LAST_EVENT:-1..2@33,LAST_EVENT_SEAT:-1..3@35,LAST_EVENT_TILE:-1..33@38,LAST_EVENT_AKA:0..1@44,"
          + "LAST_EVENT_RINSHAN:0..1@45,LAST_KAN_KIND:-1..2@46,DEFERRED_DORA:0..1@48,HAND_AKA:0..7@49,"
          + "INDICATORS:1..5@52,INDICATOR:0..33x5@55/64/70/76/82,INDICATOR_AKA:0..7@88,EVENTS:0..106@91,"
          + "LEGAL_ACTIONS:1..32@98,TRANSITIONS:0..192@103,CALL_DISCARDS:0..192@111,WINS:0..32@119}"
          + ";ObservationSchema$SeatBits{unit=32,units=1,SCORE:-32768..32767@0,RIICHI:0..2@16,IPPATSU:0..1@18}"
          + ";ObservationSchema$HandBits{unit=64,units=2,COUNT:0..4x34@0/3/6/9/12/15/18/21/24/27/30/33/36/39/42/45/"
          + "48/51/54/57/60/64/67/70/73/76/79/82/85/88/91/94/97/100}"
          + ";ObservationSchema$EventBits{unit=16,units=1,KIND:0..1@0,AUTHOR:0..3@1}"
          + ";ObservationSchema$DiscardBits{unit=16,units=1,TILE:0..33@3,AKA:0..1@9,TSUMOGIRI:0..1@10,RIICHI:0..1@11,"
          + "CALLED_BY:-1..3@12}"
          + ";ObservationSchema$MeldBits{unit=16,units=1,MELD:0..8159@3}"
          + ";ObservationSchema$LegalBits{unit=16,units=1,ACTION:0..371@0,CALL_DISCARDS:0..11@9}"
          + ";ObservationSchema$CallDiscardBits{unit=16,units=1,ACTION:0..371@0}"
          + ";ObservationSchema$WinBits{unit=64,units=2,HAN:0..75@0,FU_CODE:0..11@7,YAKUMAN:0..12@11,PAO:0..1@15,"
          + "URA_ELIGIBLE:0..1@16,URA_INDICATORS:0..5@17,SETTLEMENT:0..1@20,RANK:0..3x4@21/23/25/27,"
          + "PAYMENT:-32768..32767x4@29/45/64/80}"
          + ";ObservationSchema$EventKind=[DISCARD, MELD]"
          + ";EngineDecisionKind=[TURN, RESPONSE, CHANKAN, POST_CALL_DAHAI]"
          + ";ObservationSchema$LastEvent=[DRAW, DISCARD, KAN_ATTEMPT]"
          + ";TurnEvent$KanKind=[ANKAN, KAKAN, DAIMINKAN]"
          + ";RiichiState=[NONE, RIICHI, DOUBLE_RIICHI]"
          + ";WinSettlementProjection$SettlementAssumption=[EXACT_TSUMO, SOLE_RON]"
          + ";Meld$Type=[CHI, PON, DAIMINKAN, ANKAN, KAKAN]"
          + ";Meld$RelativeSource=[NONE, SHIMOCHA, TOIMEN, KAMICHA]"
          + ";Meld$AkaSource=[NONE, CALLED_TILE, CONSUMED_HAND_TILE, ADDED_KAN_TILE]"
          + ";meldIds=8160"
          + ";Action$Type=[DAHAI, CHI, PON, DAIMINKAN, ANKAN, KAKAN, RIICHI_DAHAI, TSUMO_AGARI, RON_AGARI, PASS,"
          + " KYUSHU_KYUHAI]"
          + ";Action$TileSelection=[NOT_APPLICABLE, NON_AKA, AKA, TSUMOGIRI, TSUMOGIRI_AKA]"
          + ";actions=372;tiles=34;pointUnit=100";

  @Test
  public void fourPlayerFingerprintIsTheReaderContract() {
    ObservationSchema schema = ObservationSchema.of(Variant.TENHOU_4P);
    assertEquals(schema.describe().toString(), FOUR_PLAYER_DESCRIPTION);
    assertEquals(schema.fingerprint(), FOUR_PLAYER_FINGERPRINT);
  }

  @Test
  public void threePlayerWidthsComeFromTheVariant() {
    ObservationSchema schema = ObservationSchema.of(Variant.TENHOU_3P);
    assertEquals(schema.variant(), Variant.TENHOU_3P);
    assertEquals(schema.header.max(HeaderBits.OBSERVER), 2);
    assertEquals(schema.header.max(HeaderBits.KYOKU), 8);
    assertEquals(schema.header.max(HeaderBits.KYOTAKU), 105);
    assertEquals(schema.header.max(HeaderBits.WALL), 55);
    assertEquals(schema.header.max(HeaderBits.EVENTS), 87);
    assertEquals(schema.header.max(HeaderBits.DECISION_KIND), 4);
    assertEquals(schema.event.max(EventBits.AUTHOR), 2);
    assertEquals(schema.legal.max(LegalBits.ACTION), 372);
    assertEquals(schema.win.count(WinBits.RANK), 3);
    assertEquals(schema.win.max(WinBits.RANK), 2);
    assertEquals(schema.win.count(WinBits.PAYMENT), 3);
    assertEquals(schema.eventKinds.constants(), List.of(EventKind.DISCARD, EventKind.MELD, EventKind.NUKI));
    assertEquals(schema.lastEvents.size(), LastEvent.values().length);
    assertEquals(ObservationSchema.of(Variant.TENHOU_4P).eventKinds.constants(), List.of(EventKind.DISCARD, EventKind.MELD));
    assertNotEquals(schema.fingerprint(), FOUR_PLAYER_FINGERPRINT);
  }

  /** フィンガープリントから、その観測のルールを引ける。どのルールとも合わなければ空を返す。 */
  @Test
  public void everyVariantIsFoundByItsFingerprint() {
    for (Variant variant : Variant.values()) {
      assertEquals(ObservationSchema.variantOf(ObservationSchema.of(variant).fingerprint()), Optional.of(variant));
    }
    assertEquals(ObservationSchema.variantOf(FOUR_PLAYER_FINGERPRINT), Optional.of(Variant.TENHOU_4P));
    assertEquals(ObservationSchema.variantOf("0123456789abcdef"), Optional.empty());
    assertEquals(ObservationSchema.variantOf(""), Optional.empty());
  }

  /** 公開する上限の定数は、観測のフィールドの範囲と同じ値。 */
  @Test
  public void publishedLimitsAreTheFieldRanges() {
    assertEquals(ObservationSchema.MAX_FU_CODE, 11);
    assertEquals(ObservationSchema.MAX_YAKUMAN, 12);
    assertEquals(ObservationSchema.MAX_CALL_DISCARDS, 11);
    for (Variant variant : Variant.values()) {
      ObservationSchema schema = ObservationSchema.of(variant);
      assertEquals(schema.win.max(WinBits.FU_CODE), ObservationSchema.MAX_FU_CODE, variant.toString());
      assertEquals(schema.win.max(WinBits.YAKUMAN), ObservationSchema.MAX_YAKUMAN, variant.toString());
      assertEquals(schema.legal.max(LegalBits.CALL_DISCARDS), ObservationSchema.MAX_CALL_DISCARDS, variant.toString());
    }
  }

  /** 110 符を超える手は、3 翻以上なら 110 符と同じ番号にまとめる（払いは変わらない）。3 翻未満なら例外を投げる。 */
  @Test
  public void foldsFuAboveTheTableAtThreeHan() {
    assertEquals(ObservationSchema.encodeFu(4, 120), 11);
    assertEquals(ObservationSchema.encodeFu(3, 110), 11);
    assertEquals(ObservationSchema.encodeFu(1, 30), 3);
    assertEquals(ObservationSchema.encodeFu(2, 25), 2);
    assertEquals(ObservationSchema.encodeFu(1, 20), 1);
    assertEquals(ObservationSchema.encodeFu(0, 0), 0);
    expectThrows(IllegalArgumentException.class, () -> ObservationSchema.encodeFu(2, 120));
    expectThrows(IllegalArgumentException.class, () -> ObservationSchema.encodeFu(1, 35));
  }
}
