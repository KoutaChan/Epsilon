package com.epsilon.agent.observe.codec;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * フィールドの置き方（宣言の順に並べる、単位をまたがない、繰り返すフィールド、前のレイアウトに続けて置く）を確かめる。範囲を外れる
 * 値を拒むこと、{@code long} 1 つで読み書きしても配列で読み書きしても同じ値になること、フィンガープリントの元になる記述も確かめる。
 */
public class BitLayoutTest {

  /** テスト用のフィールド。席の数で幅が変わるフィールドを含む。 */
  enum Head implements BitField {
    /** 席の番号。四人打ちも三人打ちも 2 bit。 */
    SEAT {
      @Override
      public int max(Variant variant) {
        return variant.seats() - 1;
      }
    },
    /** 値が無いときは −1。 */
    OPTIONAL {
      @Override
      public int min(Variant variant) {
        return -1;
      }

      @Override
      public int max(Variant variant) {
        return 6;
      }
    }
  }

  enum Body implements BitField {
    /** 13 bit。先に並ぶ {@link Head} の 5 bit と同じ単位には収まらないので、次の単位に置く。 */
    WIDE {
      @Override
      public int max(Variant variant) {
        return (1 << 13) - 1;
      }
    },
    /** 席ごとに 1 bit。 */
    FLAGS {
      @Override
      public int max(Variant variant) {
        return 1;
      }

      @Override
      public int count(Variant variant) {
        return variant.seats();
      }
    }
  }

  @Test
  public void fieldsPackInDeclarationOrderWithinUnits() {
    BitLayout<Head> head = BitLayout.of(Head.class, Short.SIZE, Variant.TENHOU_4P);
    Assert.assertEquals(head.bits(Head.SEAT), 2);
    Assert.assertEquals(head.bits(Head.OPTIONAL), 3);
    Assert.assertEquals(head.units(), 1);
    Assert.assertEquals(
        BitLayout.following(head, Body.class).describe(),
        "BitLayoutTest$Body{unit=16,units=3,WIDE:0..8191@16,FLAGS:0..1x4@29/30/31/32}");
    BitLayout<Head> threeHead = BitLayout.of(Head.class, Short.SIZE, Variant.TENHOU_3P);
    Assert.assertEquals(
        BitLayout.following(threeHead, Body.class).describe(),
        "BitLayoutTest$Body{unit=16,units=2,WIDE:0..8191@16,FLAGS:0..1x3@29/30/31}");
    Assert.assertThrows(IllegalArgumentException.class, () -> BitLayout.of(Body.class, Byte.SIZE, Variant.TENHOU_4P));
  }

  @Test
  public void valuesRoundTripAndStayInRange() {
    BitLayout<Head> head = BitLayout.of(Head.class, Short.SIZE, Variant.TENHOU_4P);
    long[] words = new long[head.longs(5) + 1];
    for (int record = 0; record < 5; record++) {
      head.put(words, 1, record, Head.SEAT, record % 4);
      head.put(words, 1, record, Head.OPTIONAL, record - 1);
    }
    for (int record = 0; record < 5; record++) {
      Assert.assertEquals(head.get(words, 1, record, Head.SEAT), record % 4);
      Assert.assertEquals(head.get(words, 1, record, Head.OPTIONAL), record - 1);
      long word = head.record(words, 1, record);
      Assert.assertEquals(head.get(word, Head.OPTIONAL), record - 1);
      Assert.assertEquals(head.with(head.with(0L, Head.SEAT, record % 4), Head.OPTIONAL, record - 1), word);
    }
    Assert.assertEquals(words[0], 0L);
    Assert.assertEquals(head.get(new long[1], 0, 0, Head.OPTIONAL), -1);
    Assert.assertThrows(IllegalArgumentException.class, () -> head.put(words, 1, 0, Head.SEAT, 4));
    Assert.assertThrows(IllegalArgumentException.class, () -> head.put(words, 1, 0, Head.OPTIONAL, -2));
    Assert.assertThrows(IllegalArgumentException.class, () -> head.putRecord(words, 1, 0, 1L << Short.SIZE));
  }

  @Test
  public void repeatedFieldsReadByIndex() {
    BitLayout<Body> body = BitLayout.of(Body.class, Long.SIZE, Variant.TENHOU_4P);
    long[] words = new long[body.longs(2)];
    body.put(words, 0, 1, Body.FLAGS, 2, 1);
    int[] flags = new int[4];
    body.getAll(words, 0, 1, Body.FLAGS, flags);
    Assert.assertEquals(flags, new int[] {0, 0, 1, 0});
    Assert.assertThrows(IllegalArgumentException.class, () -> body.get(words, 0, 1, Body.FLAGS));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> body.get(words, 0, 1, Body.FLAGS, 4));
  }

  /** 語彙が全ての定数を持つなら、その文字列は enum の {@code values()} を文字列にしたものと同じ。名前はパッケージを除いた Outer$Inner。 */
  @Test
  public void digestWritesVocabulariesLikeEnumValues() {
    SchemaDigest digest = SchemaDigest.named("test").values(Vocabulary.of(Head.class, Variant.TENHOU_4P));
    Assert.assertEquals(digest.toString(), "test;BitLayoutTest$Head=" + Arrays.toString(Head.values()));
    Assert.assertEquals(digest.hex().length(), 16);
    SchemaDigest three = SchemaDigest.named("test").values(Vocabulary.of(Head.class, Variant.TENHOU_3P));
    Assert.assertEquals(digest.hex(), three.hex());
    Assert.assertNotEquals(digest.hex(), SchemaDigest.named("test").with("x", 1).hex());
  }
}
