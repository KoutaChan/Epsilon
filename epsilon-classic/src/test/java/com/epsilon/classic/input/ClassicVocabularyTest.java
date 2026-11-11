package com.epsilon.classic.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;

import com.epsilon.classic.input.ClassicFields.TileFlag;
import com.epsilon.classic.input.ClassicVocabulary.ActionBlock;
import com.epsilon.classic.input.ClassicVocabulary.ActionGroup;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.ClassicVocabulary.DiscardContext;
import com.epsilon.classic.input.ClassicVocabulary.DrawSource;
import com.epsilon.classic.input.ClassicVocabulary.EventType;
import com.epsilon.classic.input.ClassicVocabulary.KanKind;
import com.epsilon.classic.input.ClassicVocabulary.MeldAkaSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldType;
import com.epsilon.classic.input.ClassicVocabulary.NextStep;
import com.epsilon.classic.input.ClassicVocabulary.RiichiStatus;
import com.epsilon.classic.input.ClassicVocabulary.RonFuritenKind;
import com.epsilon.classic.input.ClassicVocabulary.SettlementAssumption;
import com.epsilon.classic.input.ClassicVocabulary.TileSelection;
import com.epsilon.classic.input.ClassicVocabulary.TransitionKind;
import com.epsilon.classic.input.ClassicVocabulary.WaitWinType;
import com.epsilon.classic.input.ClassicVocabulary.WinConditions;
import com.epsilon.classic.input.ClassicVocabulary.Yaku;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.ActionSection;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/** classic の語彙の名前と並び、行動の番号のブロックの開始位置、エンジンの値からの変換が変わらないことを確かめる。 */
public class ClassicVocabularyTest {

  @Test
  public void vocabulariesKeepTheirOrder() {
    assertOrder(
        ActionType.class,
        "[DAHAI, CHI, PON, DAIMINKAN, ANKAN, KAKAN, RIICHI_DAHAI, TSUMO_AGARI, RON_AGARI, PASS, KYUSHU_KYUHAI]");
    assertOrder(ActionGroup.class, "[DAHAI, MELD, KAN, RIICHI, TSUMO, RON, PASS, KYUSHU]");
    assertOrder(TileSelection.class, "[NOT_APPLICABLE, NON_AKA, AKA, TSUMOGIRI, TSUMOGIRI_AKA]");
    assertOrder(MeldType.class, "[CHI, PON, DAIMINKAN, ANKAN, KAKAN]");
    assertOrder(MeldSource.class, "[NONE, SHIMOCHA, TOIMEN, KAMICHA]");
    assertOrder(MeldAkaSource.class, "[NONE, CALLED_TILE, CONSUMED_HAND_TILE, ADDED_KAN_TILE]");
    assertOrder(RiichiStatus.class, "[NONE, RIICHI, DOUBLE_RIICHI]");
    assertOrder(KanKind.class, "[ANKAN, KAKAN, DAIMINKAN]");
    assertOrder(DrawSource.class, "[WALL, RINSHAN]");
    assertOrder(NextStep.class, "[DECISION_COMPLETE, IMMEDIATE_DISCARD, RINSHAN_DRAW, ROUND_COMPLETE]");
    assertOrder(SettlementAssumption.class, "[EXACT_TSUMO, SOLE_RON]");
    assertOrder(EventType.class, "[NONE, DRAW, DISCARD, KAN_ATTEMPT]");
    assertOrder(
        WinConditions.class, "[NONE, TENHOU, CHIIHOU, TSUMO, RINSHAN, HAITEI, RON, CHANKAN, HOUTEI]");
    assertOrder(TransitionKind.class, "[IDENTITY, DISCARD, RINSHAN_PENDING, TERMINAL]");
    assertOrder(DiscardContext.class, "[NONE, TURN, AFTER_CHI, AFTER_PON]");
    assertOrder(RonFuritenKind.class, "[NONE, TEMPORARY, RIICHI_PERSISTENT]");
    assertOrder(WaitWinType.class, "[RON, TSUMO]");
    assertOrder(
        Yaku.class,
        "[RIICHI, DOUBLE_RIICHI, IPPATSU, MENZEN_TSUMO, PINFU, TANYAO, YAKUHAI_HAKU, YAKUHAI_HATSU, YAKUHAI_CHUN,"
            + " YAKUHAI_SEAT, YAKUHAI_ROUND, RINSHAN, CHANKAN, HAITEI, HOUTEI, IPEIKOU, SANSHOKU, SANSHOKU_DOUKOU,"
            + " ITTSU, TOITOI, SANANKOU, SANKANTSU, SHOUSANGEN, HONROUTOU, CHANTA, RYANPEIKOU, JUNCHAN, HONITSU,"
            + " CHINITSU, CHIITOITSU, KOKUSHI, SUUANKOU, CHUUREN, DAISANGEN, TSUUIISOU, RYUUIISOU, CHINROUTOU,"
            + " DAISUUSHII, SHOUSUUSHII, SUUKANTSU, TENHOU, CHIIHOU]");
    assertOrder(
        TileFlag.class,
        "[SHAPE_WAIT, RON_YAKU_WAIT, TSUMO_YAKU_WAIT, DISCARDED, CALLED_INTO_MELD, AKA_WINNING_TILE]");
  }

  @Test
  public void actionTypesDeclareTheirGroupAndNextStep() {
    List<String> declared = new ArrayList<>();
    for (ActionType type : ActionType.values()) {
      declared.add(type + ":" + type.group() + ":" + type.next());
    }
    assertEquals(
        declared,
        List.of(
            "DAHAI:DAHAI:DECISION_COMPLETE",
            "CHI:MELD:IMMEDIATE_DISCARD",
            "PON:MELD:IMMEDIATE_DISCARD",
            "DAIMINKAN:MELD:RINSHAN_DRAW",
            "ANKAN:KAN:RINSHAN_DRAW",
            "KAKAN:KAN:RINSHAN_DRAW",
            "RIICHI_DAHAI:RIICHI:DECISION_COMPLETE",
            "TSUMO_AGARI:TSUMO:ROUND_COMPLETE",
            "RON_AGARI:RON:ROUND_COMPLETE",
            "PASS:PASS:DECISION_COMPLETE",
            "KYUSHU_KYUHAI:KYUSHU:ROUND_COMPLETE"));
  }

  @Test
  public void actionBlocksKeepTheirStarts() {
    List<String> starts = new ArrayList<>();
    for (ActionBlock block : ActionBlock.values()) {
      starts.add(block + ":" + block.start());
    }
    assertEquals(
        starts,
        List.of(
            "DAHAI:0",
            "DAHAI_TSUMOGIRI:34",
            "DAHAI_AKA:68",
            "DAHAI_TSUMOGIRI_AKA:71",
            "CHI:74",
            "CHI_AKA:137",
            "PON:155",
            "PON_AKA:189",
            "DAIMINKAN:192",
            "ANKAN:226",
            "KAKAN:260",
            "RIICHI:294",
            "RIICHI_TSUMOGIRI:328",
            "RIICHI_AKA:362",
            "RIICHI_TSUMOGIRI_AKA:365",
            "TSUMO:368",
            "RON:369",
            "PASS:370",
            "KYUSHU_KYUHAI:371"));
    assertEquals(ClassicVocabulary.ACTION_SPACE, 372);
    assertEquals(ClassicVocabulary.DISCARD_IDENTITIES, 74);
  }

  /** 四人打ちの行動の番号は classic の行動の番号と同じ並びで、どの行動も同じ番号に変換される。 */
  @Test
  public void everyFourPlayerActionKeepsItsNumber() {
    int space = ActionSection.space(Variant.TENHOU_4P);
    assertEquals(space, ClassicVocabulary.ACTION_SPACE);
    for (int code = 0; code < space; code++) {
      Action action = Action.of(code);
      assertEquals(VocabularyCodec.actionIndex(action), code, action.toString());
      assertEquals(VocabularyCodec.actionType(action.type()).name(), action.type().name());
      assertEquals(VocabularyCodec.tileSelection(action.selection()).name(), action.selection().name());
      int identity = VocabularyCodec.discardIdentity(action);
      switch (action.type()) {
        case DAHAI -> assertEquals(identity, code);
        case RIICHI_DAHAI -> assertEquals(identity, code - ActionBlock.RIICHI.start());
        default -> assertEquals(identity, -1);
      }
    }
    Action nuki = Action.of(Action.code(Action.Type.NUKI));
    assertThrows(IllegalArgumentException.class, () -> VocabularyCodec.actionIndex(nuki));
  }

  @Test
  public void yakuMapByName() {
    for (ScoringYaku yaku : ScoringYaku.values()) {
      assertEquals(VocabularyCodec.yaku(yaku.bit()), 1L << Yaku.valueOf(yaku.name()).ordinal(), yaku.name());
    }
    assertEquals(VocabularyCodec.yaku(0), 0);
  }

  @Test
  public void tileCodesPackCountsAndFlags() {
    assertEquals(TileFlag.CODES, 1601);
    assertEquals(TileFlag.code(0, 0, 0), 1);
    assertEquals(TileFlag.code(4, 4, 63), 1600);
    assertEquals(TileFlag.SHAPE_WAIT.bit(), 32);
    assertEquals(TileFlag.AKA_WINNING_TILE.bit(), 1);
  }

  @Test
  public void yakuChunksAndFuCodesFollowTheirRules() {
    assertEquals(ClassicSchema.YAKU_CHUNKS, 3);
    long yaku = 1L << Yaku.CHIIHOU.ordinal() | 1L << Yaku.RIICHI.ordinal();
    assertEquals(ClassicTokenizer.yakuChunk(yaku, 0), 2);
    assertEquals(ClassicTokenizer.yakuChunk(yaku, 1), 1);
    assertEquals(ClassicTokenizer.yakuChunk(yaku, 2), (1 << (Yaku.CHIIHOU.ordinal() - 28)) + 1);
    assertEquals(ClassicTokenizer.fuCode(0), 0);
    assertEquals(ClassicTokenizer.fuCode(20), 1);
    assertEquals(ClassicTokenizer.fuCode(25), 2);
    assertEquals(ClassicTokenizer.fuCode(40), 4);
    assertEquals(ClassicTokenizer.fuCode(130), 11);
  }

  @Test
  public void drawHitProbabilitiesDrawWithoutReplacement() {
    assertEquals(ClassicTokenizer.hit(0, 100, 3), 0f, 0f);
    assertEquals(ClassicTokenizer.hit(10, 100, 1), 0.1f, 1e-6f);
    assertEquals(ClassicTokenizer.hit(10, 100, 2), 1 - 90f / 100 * 89 / 99, 1e-6f);
    assertEquals(ClassicTokenizer.hit(2, 2, 1), 1f, 0f);
  }

  private static void assertOrder(Class<? extends Enum<?>> type, String expected) {
    assertEquals(Arrays.toString(type.getEnumConstants()), expected, type.getSimpleName());
  }
}
