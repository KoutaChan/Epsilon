package com.epsilon.prism.input;

import org.testng.Assert;
import org.testng.annotations.Test;

public class CodeSpaceTest {
  private enum Place {
    HIGH,
    LOW
  }

  @Test
  public void partsPartitionTheSpaceInDeclarationOrder() {
    var space = new CodeSpace<>(Place.class, digit -> digit == Place.HIGH ? 3 : 5);
    Assert.assertEquals(space.describe(), "HIGH=3,LOW=5");
    Assert.assertEquals(space.size(), 15);
    for (int code = 0; code < space.size(); code++) {
      int high = space.digit(code, Place.HIGH);
      int low = space.digit(code, Place.LOW);
      Assert.assertEquals(high, code / 5);
      Assert.assertEquals(low, code % 5);
      Assert.assertEquals(space.part(Place.HIGH, high) + space.part(Place.LOW, low), code);
    }
    Assert.expectThrows(IndexOutOfBoundsException.class, () -> space.part(Place.HIGH, 3));
    Assert.expectThrows(IndexOutOfBoundsException.class, () -> space.digit(15, Place.LOW));
    Assert.expectThrows(IllegalArgumentException.class, () -> new CodeSpace<>(Place.class, d -> 0));
    Assert.expectThrows(ArithmeticException.class, () -> new CodeSpace<>(Place.class, d -> 65536));
  }

  @Test
  public void selectedDigitsDetermineTheEncodingOrder() {
    var selection = new Place[] {Place.LOW, Place.HIGH};
    var space = new CodeSpace<>(Place.class, selection, d -> d == Place.LOW ? 5 : 3);
    selection[0] = Place.HIGH;
    Assert.assertEquals(space.encode(3, 2), 11);
    Assert.assertEquals(space.describe(), "LOW=5,HIGH=3");
    Assert.assertEquals(space.digit(11, Place.LOW), 3);
    var single = new CodeSpace<>(Place.class, new Place[] {Place.LOW}, d -> 5);
    Assert.expectThrows(IllegalArgumentException.class, () -> single.digit(0, Place.HIGH));
    Assert.expectThrows(IllegalArgumentException.class, () -> single.encode(0, 0));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new CodeSpace<>(Place.class, new Place[] {Place.LOW, Place.LOW}, d -> 2));
  }
}
