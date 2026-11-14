package com.epsilon.mahjong.io.majsoul;

import com.epsilon.mahjong.record.RecordedGame;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 雀魂の牌譜を取る API の応答（protobuf）を読むテスト。{@link MajSoulReaderTest#RECORD} と同じ半荘を、liqi.proto のフィールド番号で
 * protobuf に書いて使う。proto3 と同じく既定値（席 0、0 本場、ツモ切りでない打牌など）は書かない。局の始まりの点は詰めた形、和了の
 * 点の動きは詰めない形で書き、負の int32 を含む。
 */
public class MajSoulProtoTest {

  /** GameDetailRecords の version。記録を actions に置く形の版。 */
  private static final int ACTIONS_VERSION = 210_715;

  /**
   * 応答は、記録を actions の result に置く形（記録でない操作の action を挟む）でも、records に置く形でも、同じ半荘の JSON と同じ半荘に
   * 読む。
   */
  @Test
  public void aResponseReadsLikeTheSameGamesJson() {
    RecordedGame json = MajSoulReader.read(MajSoulReaderTest.RECORD, "soul");
    Assert.assertTrue(MajSoulReader.isResponse(response(true)));
    Assert.assertEquals(MajSoulReader.read(response(true), "soul"), json);
    Assert.assertEquals(MajSoulReader.read(response(false), "soul"), json);
    Assert.assertFalse(MajSoulReader.isResponse(MajSoulReaderTest.RECORD.getBytes(StandardCharsets.UTF_8)));
  }

  /** 途中で切れた応答と、data の無い応答は、IllegalArgumentException を投げて読まない。 */
  @Test
  public void brokenResponsesAreRejected() {
    byte[] whole = response(true);
    byte[] cut = Arrays.copyOf(whole, whole.length - 7);
    Assert.assertThrows(IllegalArgumentException.class, () -> MajSoulReader.read(cut, "cut"));
    byte[] headOnly = wrapper(".lq.ResGameRecord", new Proto().message(3, head())).bytes();
    IllegalArgumentException failure =
        Assert.expectThrows(IllegalArgumentException.class, () -> MajSoulReader.read(headOnly, "head-only"));
    Assert.assertTrue(failure.getMessage().contains("no data"), failure.getMessage());
  }

  /**
   * {@link MajSoulReaderTest#RECORD} と同じ半荘を、牌譜を取る API の応答（ResGameRecord を包んだ lq.Wrapper）にする。
   *
   * @param actions true なら記録を actions の result に置き、false なら records に置く
   */
  public static byte[] response(boolean actions) {
    List<Proto> records =
        List.of(
            wrapper(
                ".lq.RecordNewRound",
                new Proto()
                    .varint(1, 1)
                    .varint(2, 3)
                    .packed(5, 24_000, 23_000, 29_000, 24_000)
                    .strings(7, "2p", "4p", "2m", "8m", "2s", "8s", "4z", "4z", "5z", "5z", "6z", "7z", "7z")
                    .strings(8, "3p", "3p", "3m", "3m", "7m", "7m", "3s", "3s", "6s", "7s", "7s", "6z", "6z")
                    .strings(9, "2m", "3m", "4m", "5m", "6m", "7m", "3p", "4p", "5p", "6s", "6s", "7s", "8s")
                    .strings(10, "1m", "9m", "1p", "9p", "1s", "9s", "1z", "1z", "2z", "2z", "3z", "3p", "7z", "8p")
                    .strings(16, "9s")),
            wrapper(".lq.RecordDiscardTile", new Proto().varint(1, 3).string(2, "3p")),
            wrapper(
                ".lq.RecordChiPengGang",
                new Proto().varint(1, 1).varint(2, 1).strings(3, "3p", "3p", "3p").packed(4, 1, 1, 3)),
            wrapper(".lq.RecordDiscardTile", new Proto().varint(1, 1).string(2, "6s")),
            wrapper(
                ".lq.RecordHule",
                new Proto()
                    .message(
                        1,
                        new Proto()
                            .string(3, "6s")
                            .varint(4, 2)
                            .varint(11, 2)
                            .message(12, new Proto().varint(2, 1).varint(3, 12))
                            .message(12, new Proto().varint(2, 1).varint(3, 14))
                            .varint(13, 30)
                            .varint(15, 2_000))
                    .varint(3, 0)
                    .varint(3, -2_000)
                    .varint(3, 2_000)
                    .varint(3, 0)));
    Proto details = new Proto();
    if (actions) {
      details.varint(2, ACTIONS_VERSION);
      for (Proto record : records) {
        details.message(3, new Proto().varint(2, 1).message(3, record));
        details.message(3, new Proto().varint(2, 2).bytes(4, new byte[] {8, 1}));
      }
    } else {
      for (Proto record : records) {
        details.message(1, record);
      }
    }
    Proto response = new Proto().message(3, head()).message(4, wrapper(".lq.GameDetailRecords", details));
    return wrapper(".lq.ResGameRecord", response).bytes();
  }

  /** 応答の head（RecordGame）。打ち手の名前は席 0 と席 2 だけにあり、終局の点は席の順でない。 */
  private static Proto head() {
    return new Proto()
        .string(1, "soul")
        .message(11, new Proto().varint(2, 2).string(3, "West"))
        .message(11, new Proto().string(3, "East"))
        .message(11, new Proto().varint(2, 1))
        .message(
            12,
            new Proto()
                .message(1, new Proto().varint(1, 2).varint(3, 31_000))
                .message(1, new Proto().varint(3, 24_000))
                .message(1, new Proto().varint(1, 1).varint(3, 21_000))
                .message(1, new Proto().varint(1, 3).varint(3, 24_000)));
  }

  private static Proto wrapper(String name, Proto data) {
    return new Proto().string(1, name).message(2, data);
  }

  /** テストで protobuf のメッセージを書く。 */
  private static final class Proto {

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    Proto varint(int field, long value) {
      tag(field, 0);
      raw(value);
      return this;
    }

    /** 繰り返しの数を詰めた形で書く。 */
    Proto packed(int field, long... values) {
      Proto packed = new Proto();
      for (long value : values) {
        packed.raw(value);
      }
      return bytes(field, packed.bytes());
    }

    Proto string(int field, String value) {
      return bytes(field, value.getBytes(StandardCharsets.UTF_8));
    }

    Proto strings(int field, String... values) {
      for (String value : values) {
        string(field, value);
      }
      return this;
    }

    Proto message(int field, Proto value) {
      return bytes(field, value.bytes());
    }

    Proto bytes(int field, byte[] value) {
      tag(field, 2);
      raw(value.length);
      out.writeBytes(value);
      return this;
    }

    byte[] bytes() {
      return out.toByteArray();
    }

    private void tag(int field, int type) {
      raw((long) field << 3 | type);
    }

    private void raw(long value) {
      long rest = value;
      while ((rest & ~0x7FL) != 0) {
        out.write((int) (rest & 0x7F) | 0x80);
        rest >>>= 7;
      }
      out.write((int) rest);
    }
  }
}
