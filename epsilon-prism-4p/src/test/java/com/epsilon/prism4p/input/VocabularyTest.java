package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 語彙の契約のテスト。四人打ちの語彙の並びが、テストに書いた名前の並びと一致する。入力に書く値は語彙の位置 + 1 なので、並びが変わると
 * 書く値も変わる。値が位置 + 1 になることと、四人打ちの語彙に無い北抜きの定数を拒むことも確かめる。手の種類ごとの価値グラフでの扱いは、
 * テストに書いた対応のとおりになる。
 */
public class VocabularyTest {

  @Test
  public void fourPlayerVocabulariesKeepTheOrder() {
    assertEquals(
        names(Action.Type.class),
        List.of(
            "DAHAI", "CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN", "RIICHI_DAHAI", "TSUMO_AGARI", "RON_AGARI", "PASS",
            "KYUSHU_KYUHAI"));
    assertEquals(names(EngineDecisionKind.class), List.of("TURN", "RESPONSE", "CHANKAN", "POST_CALL_DAHAI"));
    assertEquals(
        names(Action.TileSelection.class), List.of("NOT_APPLICABLE", "NON_AKA", "AKA", "TSUMOGIRI", "TSUMOGIRI_AKA"));
    assertEquals(names(TurnEvent.KanKind.class), List.of("ANKAN", "KAKAN", "DAIMINKAN"));
    assertEquals(names(RiichiState.class), List.of("NONE", "RIICHI", "DOUBLE_RIICHI"));
    assertEquals(names(Meld.Type.class), List.of("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN"));
    assertEquals(names(Meld.RelativeSource.class), List.of("NONE", "SHIMOCHA", "TOIMEN", "KAMICHA"));
    assertEquals(names(Meld.AkaSource.class), List.of("NONE", "CALLED_TILE", "CONSUMED_HAND_TILE", "ADDED_KAN_TILE"));
    assertEquals(names(SettlementAssumption.class), List.of("EXACT_TSUMO", "SOLE_RON"));
  }

  @Test
  public void idsArePositionPlusOneAndRejectNuki() {
    Vocabulary<Action.Type> types = Vocabulary.of(Action.Type.class, TENHOU_4P);
    for (int index = 0; index < types.size(); index++) {
      assertEquals(Ids.of(types, types.get(index)), index + 1);
    }
    assertEquals(Ids.of(types, null), Ids.NONE);
    expectThrows(IllegalArgumentException.class, () -> Ids.of(types, Action.Type.NUKI));
    Vocabulary<EngineDecisionKind> kinds = Vocabulary.of(EngineDecisionKind.class, TENHOU_4P);
    expectThrows(IllegalArgumentException.class, () -> Ids.of(kinds, EngineDecisionKind.NUKI_RESPONSE));
  }

  @Test
  public void actionRolesFollowTheTypes() {
    StringBuilder roles = new StringBuilder();
    for (Action.Type type : Vocabulary.of(Action.Type.class, TENHOU_4P).constants()) {
      roles.append(type).append("->").append(ActionRole.of(type)).append(' ');
    }
    assertEquals(
        roles.toString().trim(),
        "DAHAI->DISCARD CHI->CALL PON->CALL DAIMINKAN->CALL ANKAN->KAN KAKAN->KAN RIICHI_DAHAI->RIICHI"
            + " TSUMO_AGARI->TSUMO RON_AGARI->RON PASS->PASS KYUSHU_KYUHAI->KYUSHU");
    expectThrows(IllegalArgumentException.class, () -> ActionRole.of(Action.Type.NUKI));
  }

  private static <E extends Enum<E>> List<String> names(Class<E> type) {
    List<String> names = new ArrayList<>();
    for (E constant : Vocabulary.of(type, TENHOU_4P).constants()) {
      names.add(constant.name());
    }
    return names;
  }
}
