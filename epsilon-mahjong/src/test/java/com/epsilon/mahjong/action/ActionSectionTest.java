package com.epsilon.mahjong.action;

import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 行動の番号を分けるセクションを確かめる。各セクションの先頭と幅は系譜の記録に残っているので、変えない。 */
public class ActionSectionTest {

  /** セクションごとの先頭の番号。 */
  private static final Map<ActionSection, Integer> STARTS = new EnumMap<>(ActionSection.class);

  static {
    STARTS.put(ActionSection.DAHAI, 0);
    STARTS.put(ActionSection.DAHAI_TSUMOGIRI, 34);
    STARTS.put(ActionSection.DAHAI_AKA, 68);
    STARTS.put(ActionSection.DAHAI_TSUMOGIRI_AKA, 71);
    STARTS.put(ActionSection.CHI, 74);
    STARTS.put(ActionSection.CHI_AKA, 137);
    STARTS.put(ActionSection.PON, 155);
    STARTS.put(ActionSection.PON_AKA, 189);
    STARTS.put(ActionSection.DAIMINKAN, 192);
    STARTS.put(ActionSection.ANKAN, 226);
    STARTS.put(ActionSection.KAKAN, 260);
    STARTS.put(ActionSection.RIICHI_DAHAI, 294);
    STARTS.put(ActionSection.RIICHI_TSUMOGIRI, 328);
    STARTS.put(ActionSection.RIICHI_AKA, 362);
    STARTS.put(ActionSection.RIICHI_TSUMOGIRI_AKA, 365);
    STARTS.put(ActionSection.TSUMO_AGARI, 368);
    STARTS.put(ActionSection.RON_AGARI, 369);
    STARTS.put(ActionSection.PASS, 370);
    STARTS.put(ActionSection.KYUSHU_KYUHAI, 371);
    STARTS.put(ActionSection.NUKI, 372);
  }

  @Test
  public void startsMatchTheContract() {
    Assert.assertEquals(STARTS.size(), ActionSection.values().length);
    for (ActionSection section : ActionSection.values()) {
      Assert.assertEquals(section.start(), (int) STARTS.get(section), section.name());
    }
  }

  @Test
  public void sectionsFollowEachOtherInDeclarationOrder() {
    int next = 0;
    for (ActionSection section : ActionSection.values()) {
      Assert.assertEquals(section.start(), next, section.name());
      Assert.assertEquals(section.end(), next + section.width(), section.name());
      next = section.end();
    }
    Assert.assertEquals(ActionSection.SPACE, next);
    Assert.assertEquals(ActionSection.SPACE, 373);
  }

  @Test
  public void eachVariantEndsAtItsLastSection() {
    Assert.assertEquals(ActionSection.space(Variant.TENHOU_4P), 372);
    Assert.assertEquals(ActionSection.space(Variant.TENHOU_3P), 373);
  }

  @Test
  public void codesMapBackToTheirSection() {
    for (int code = 0; code < ActionSection.SPACE; code++) {
      ActionSection section = ActionSection.of(code);
      Assert.assertTrue(section.contains(code), "code " + code);
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> ActionSection.of(-1));
    Assert.assertThrows(IllegalArgumentException.class, () -> ActionSection.of(ActionSection.SPACE));
  }

  @Test
  public void threePlayersDropOnlyTheChiSections() {
    List<ActionSection> three = new ArrayList<>();
    List<ActionSection> four = new ArrayList<>();
    for (ActionSection section : ActionSection.values()) {
      if (section.type() != Type.CHI) {
        three.add(section);
      }
      if (section.type() != Type.NUKI) {
        four.add(section);
      }
    }
    Assert.assertEquals(Vocabulary.of(ActionSection.class, Variant.TENHOU_3P).constants(), three);
    Assert.assertEquals(Vocabulary.of(ActionSection.class, Variant.TENHOU_4P).constants(), four);
    Assert.assertSame(ActionSection.CHI_AKA.requires(), RuleFlag.CHI);
    Assert.assertSame(ActionSection.NUKI.requires(), RuleFlag.NUKI);
    Assert.assertNull(ActionSection.PON_AKA.requires());
  }

  @Test
  public void typeAndSelectionNameAtMostOneSection() {
    for (Type type : Type.values()) {
      for (TileSelection selection : TileSelection.values()) {
        ActionSection found = null;
        for (ActionSection section : ActionSection.values()) {
          if (section.type() == type && section.selection() == selection) {
            Assert.assertNull(found, type + " " + selection);
            found = section;
          }
        }
        if (found == null) {
          Assert.assertThrows(IllegalArgumentException.class, () -> ActionSection.of(type, selection));
        } else {
          Assert.assertSame(ActionSection.of(type, selection), found);
        }
      }
    }
  }
}
