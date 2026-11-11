package com.epsilon.classic.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.classic.input.Buckets.Bucket;
import org.testng.annotations.Test;

/** 行の大きさから区分を選ぶ処理と、区分の番号を確かめる。 */
public class BucketsTest {

  @Test
  public void requiredPicksTheSmallestBucketsThatHoldTheRow() {
    assertEquals(Buckets.required(1, 1), new Bucket(1, 1));
    assertEquals(Buckets.required(3, 1), new Bucket(4, 1));
    assertEquals(Buckets.required(9, 2), new Bucket(10, 4));
    assertEquals(Buckets.required(14, 5), new Bucket(14, 8));
    assertEquals(Buckets.required(17, 9), new Bucket(32, 12));
    assertEquals(Buckets.required(33, 13), new Bucket(40, 16));
    assertEquals(Buckets.required(40, 16), new Bucket(40, 16));
  }

  @Test
  public void inferenceSharesShapesBetweenRows() {
    assertEquals(Buckets.inference(1, 1), new Bucket(1, 1));
    assertEquals(Buckets.inference(2, 1), new Bucket(16, 1));
    assertEquals(Buckets.inference(16, 1), new Bucket(16, 1));
    assertEquals(Buckets.inference(17, 1), new Bucket(40, 1));
    assertEquals(Buckets.inference(3, 2), new Bucket(4, 16));
    assertEquals(Buckets.inference(7, 11), new Bucket(8, 16));
    assertEquals(Buckets.inference(9, 4), new Bucket(16, 16));
    assertEquals(Buckets.inference(20, 3), new Bucket(40, 16));
  }

  @Test
  public void rowsOutsideTheBucketsAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> Buckets.required(0, 1));
    assertThrows(IllegalArgumentException.class, () -> Buckets.required(41, 1));
    assertThrows(IllegalArgumentException.class, () -> Buckets.required(1, 0));
    assertThrows(IllegalArgumentException.class, () -> Buckets.required(1, 17));
    assertThrows(IllegalArgumentException.class, () -> Buckets.inference(41, 1));
    assertThrows(IllegalArgumentException.class, () -> new Bucket(3, 1));
    assertThrows(IllegalArgumentException.class, () -> new Bucket(4, 2));
  }

  @Test
  public void bucketsAreSharedAndNumberedInOrder() {
    assertSame(Buckets.required(3, 1), Buckets.required(4, 1));
    assertSame(Buckets.inference(2, 1), Buckets.required(16, 1));
    assertEquals(Buckets.count(), 50);
    assertEquals(Buckets.ordinal(new Bucket(1, 1)), 0);
    assertEquals(Buckets.ordinal(new Bucket(1, 4)), 1);
    assertEquals(Buckets.ordinal(new Bucket(2, 1)), 5);
    assertEquals(Buckets.ordinal(new Bucket(40, 16)), 49);
    for (int ordinal = 0; ordinal < Buckets.count(); ordinal++) {
      assertEquals(Buckets.ordinal(Buckets.of(ordinal)), ordinal);
    }
    assertThrows(IndexOutOfBoundsException.class, () -> Buckets.of(Buckets.count()));
  }

  @Test
  public void bucketsTellWhetherARowFits() {
    Bucket bucket = new Bucket(8, 4);
    assertTrue(bucket.fits(8, 4));
    assertTrue(bucket.fits(1, 1));
    assertFalse(bucket.fits(9, 1));
    assertFalse(bucket.fits(8, 5));
  }
}
