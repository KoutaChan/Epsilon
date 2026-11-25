package com.epsilon.prism.input;

import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** ルールと機能の組ごとに、形式の同一性とキーの幅を確かめる。 */
public class InputFormatTest {
  @DataProvider
  public static Variant[] variants() {
    return Variant.values();
  }

  @DataProvider
  public static InputFormat[] formats() {
    return Arrays.stream(Variant.values())
        .flatMap(
            variant ->
                List.of(
                    InputFormat.of(variant),
                    InputFormat.of(variant, Set.of(InputFeature.PUBLIC_ORDER_EVENTS)),
                    InputFormat.of(variant, Set.of(InputFeature.CANDIDATE_SEGMENTS)),
                    InputFormat.of(variant, EnumSet.allOf(InputFeature.class)))
                    .stream())
        .toArray(InputFormat[]::new);
  }

  @Test(dataProvider = "formats")
  public void featuresFixTheCodeWidthAndFingerprint(InputFormat format) {
    var selection = EnumSet.noneOf(InputFeature.class);
    for (var feature : InputFeature.values()) if (format.has(feature)) selection.add(feature);
    Assert.assertSame(InputFormat.of(format.variant(), selection), format);
    selection.clear();
    Assert.assertSame(
        InputFormat.of(format.variant(), selection), InputFormat.of(format.variant()));
    int segments = format.has(InputFeature.CANDIDATE_SEGMENTS) ? 2 + format.seats() : 0;
    Assert.assertEquals(format.candidateCodes().slots().size(), segments);
    Assert.assertEquals(
        format.width(InputBlob.Tensor.TRANSITION_CODE),
        format.fixedTokens() + segments * format.tiles());
    Assert.assertEquals(
        Arrays.stream(formats()).map(InputFormat::fingerprint).distinct().count(),
        (long) formats().length);
  }
}
