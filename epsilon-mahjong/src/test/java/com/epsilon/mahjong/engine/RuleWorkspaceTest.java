package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 作業領域はルールの部品を自分で持ち、ほかの作業領域と何も共有しない。 */
public class RuleWorkspaceTest {

  @Test
  public void eachWorkspaceOwnsItsPartsForTheVariant() {
    for (Variant variant : Variant.values()) {
      RuleWorkspace first = new RuleWorkspace(variant);
      RuleWorkspace second = new RuleWorkspace(variant);
      Assert.assertSame(first.variant(), variant);
      Assert.assertSame(first.context().variant(), variant);
      Fields.assertSameFields(first, second);
    }
    Assert.assertThrows(NullPointerException.class, () -> new RuleWorkspace(null));
  }
}
