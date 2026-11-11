package com.epsilon.mahjong.rules;

import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** ルールで使う定数だけに絞った列の、並び・添字・文字列の形を確かめる。 */
public class VocabularyTest {

  private enum Sample implements VariantScoped {
    PLAIN,
    FOUR_ONLY(RuleFlag.CHI),
    MIDDLE,
    THREE_ONLY(RuleFlag.NUKI),
    LAST;

    private final RuleFlag requires;

    Sample() {
      this(null);
    }

    Sample(RuleFlag requires) {
      this.requires = requires;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  @Test
  public void scopedConstantsKeepTheDeclarationOrder() {
    Vocabulary<Sample> four = Vocabulary.of(Sample.class, Variant.TENHOU_4P);
    Assert.assertEquals(four.constants(), List.of(Sample.PLAIN, Sample.FOUR_ONLY, Sample.MIDDLE, Sample.LAST));
    Vocabulary<Sample> three = Vocabulary.of(Sample.class, Variant.TENHOU_3P);
    Assert.assertEquals(three.constants(), List.of(Sample.PLAIN, Sample.MIDDLE, Sample.THREE_ONLY, Sample.LAST));
    Assert.assertSame(three.type(), Sample.class);
    Assert.assertSame(three.variant(), Variant.TENHOU_3P);
  }

  @Test
  public void indexesArePositionsInTheList() {
    for (Variant variant : Variant.values()) {
      Vocabulary<Sample> vocabulary = Vocabulary.of(Sample.class, variant);
      for (int index = 0; index < vocabulary.size(); index++) {
        Assert.assertEquals(vocabulary.index(vocabulary.get(index)), index);
        Assert.assertTrue(vocabulary.contains(vocabulary.get(index)));
      }
    }
    Vocabulary<Sample> three = Vocabulary.of(Sample.class, Variant.TENHOU_3P);
    Assert.assertEquals(three.index(Sample.MIDDLE), 1);
    Assert.assertEquals(three.index(Sample.THREE_ONLY), 2);
    Assert.assertFalse(three.contains(Sample.FOUR_ONLY));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.index(Sample.FOUR_ONLY));
  }

  @Test
  public void textHasTheFormOfArraysToString() {
    Assert.assertEquals(Vocabulary.of(Sample.class, Variant.TENHOU_3P).toString(), "[PLAIN, MIDDLE, THREE_ONLY, LAST]");
    for (Variant variant : Variant.values()) {
      Vocabulary<AbortiveDraw> unscoped = Vocabulary.of(AbortiveDraw.class, variant);
      Assert.assertEquals(unscoped.constants(), List.of(AbortiveDraw.values()));
      Assert.assertEquals(unscoped.toString(), Arrays.toString(AbortiveDraw.values()));
    }
  }

  @Test
  public void listCannotBeChanged() {
    Vocabulary<Sample> vocabulary = Vocabulary.of(Sample.class, Variant.TENHOU_4P);
    Assert.assertThrows(UnsupportedOperationException.class, () -> vocabulary.constants().add(Sample.PLAIN));
  }

  @Test
  public void scopedConstantsKnowTheirVariants() {
    Assert.assertTrue(Sample.PLAIN.in(Variant.TENHOU_4P));
    Assert.assertTrue(Sample.PLAIN.in(Variant.TENHOU_3P));
    Assert.assertTrue(Sample.FOUR_ONLY.in(Variant.TENHOU_4P));
    Assert.assertFalse(Sample.FOUR_ONLY.in(Variant.TENHOU_3P));
    Assert.assertFalse(Sample.THREE_ONLY.in(Variant.TENHOU_4P));
    Assert.assertTrue(Sample.THREE_ONLY.in(Variant.TENHOU_3P));
  }
}
