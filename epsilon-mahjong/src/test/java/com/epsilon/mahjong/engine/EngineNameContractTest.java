package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PRISM が語彙に使う名前（判断の種類、槓の種類、和了の精算の仮定）を確かめる。フィンガープリントには、パッケージを除いた型の
 * 名前（Outer$Inner）と、語彙を並べた文字列が入る。語彙には、そのルールで使う定数だけを宣言の順に並べる。四人打ちの文字列は
 * 学習中の系譜のフィンガープリントに入っているので、文字列そのものを比べる。
 */
public class EngineNameContractTest {

  @Test
  public void decisionKindsKeepTheirTextAndAddNukiResponseLast() {
    Assert.assertEquals(
        Vocabulary.of(EngineDecisionKind.class, Variant.TENHOU_4P).toString(),
        "[TURN, RESPONSE, CHANKAN, POST_CALL_DAHAI]");
    Vocabulary<EngineDecisionKind> three = Vocabulary.of(EngineDecisionKind.class, Variant.TENHOU_3P);
    Assert.assertEquals(three.toString(), "[TURN, RESPONSE, CHANKAN, POST_CALL_DAHAI, NUKI_RESPONSE]");
    Assert.assertEquals(three.index(EngineDecisionKind.NUKI_RESPONSE), 4);
  }

  @Test
  public void kanKindsKeepTheirText() {
    for (Variant variant : Variant.values()) {
      Assert.assertEquals(
          Vocabulary.of(TurnEvent.KanKind.class, variant).toString(), "[ANKAN, KAKAN, DAIMINKAN]", variant.name());
    }
  }

  @Test
  public void settlementAssumptionsKeepTheirText() {
    for (Variant variant : Variant.values()) {
      Assert.assertEquals(
          Vocabulary.of(WinSettlementProjection.SettlementAssumption.class, variant).toString(),
          "[EXACT_TSUMO, SOLE_RON]",
          variant.name());
    }
  }

  @Test
  public void digestNamesKeepTheOuterClasses() {
    Assert.assertEquals(digestName(EngineDecisionKind.class), "EngineDecisionKind");
    Assert.assertEquals(digestName(TurnEvent.KanKind.class), "TurnEvent$KanKind");
    Assert.assertEquals(
        digestName(WinSettlementProjection.SettlementAssumption.class), "WinSettlementProjection$SettlementAssumption");
  }

  private static String digestName(Class<?> type) {
    return type.getName().substring(type.getPackageName().length() + 1);
  }
}
