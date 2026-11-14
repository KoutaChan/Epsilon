package com.epsilon.mahjong.io;

import com.epsilon.mahjong.io.majsoul.MajSoulProtoTest;
import com.epsilon.mahjong.io.majsoul.MajSoulReader;
import com.epsilon.mahjong.io.majsoul.MajSoulReaderTest;
import com.epsilon.mahjong.io.mjai.MjaiCodec;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜ファイルのバイト列を読む {@code RecordFiles} のテスト。gzip と BOM の扱い、形式の判定、大きさの上限、読まずに弾く中身を確かめる。 */
public class RecordFilesTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /**
   * どの形式も、その形式のリーダーと同じ半荘に読む。試すのは、自己対局の半荘を書いた天鳳の XML（gzip と、BOM と空白の付いた平文）、
   * mjai（1 行 1 イベント、配列、gzip、{@code type} が先頭のフィールドでないイベント）、雀魂の JSON と、同じ半荘の雀魂の牌譜を取る
   * API の応答の protobuf（そのままと gzip）。
   */
  @Test
  public void everyFormatReadsLikeItsOwnReader() {
    String id = "self-0";
    String xml = TenhouXml.write(RecordedGames.play(VARIANT, 0, id));
    RecordedGame tenhou = TenhouXml.read(xml, id);
    Assert.assertEquals(RecordFiles.read(gzip(utf8(xml)), id), tenhou);
    Assert.assertEquals(RecordFiles.read(utf8("\uFEFF \n" + xml + "\n"), id), tenhou);

    String mjai = MjaiCodec.write(tenhou);
    RecordedGame read = MjaiCodec.read(mjai, id);
    Assert.assertEquals(read.meta().source(), RecordedGame.Source.MJAI);
    Assert.assertEquals(RecordFiles.read(utf8(mjai), id), read);
    Assert.assertEquals(RecordFiles.read(gzip(utf8(mjai)), id), read);
    Assert.assertEquals(RecordFiles.read(utf8("[" + String.join(",", mjai.strip().split("\n")) + "]"), id), read);
    String later = mjai.replaceFirst("\\{\"type\":\"start_game\",", "{\"id\":0,\"type\":\"start_game\",");
    Assert.assertNotEquals(later, mjai);
    Assert.assertEquals(RecordFiles.read(utf8(later), id), read);

    RecordedGame soul = MajSoulReader.read(MajSoulReaderTest.RECORD, "soul");
    Assert.assertEquals(soul.meta().source(), RecordedGame.Source.MAJSOUL);
    Assert.assertEquals(RecordFiles.read(utf8(MajSoulReaderTest.RECORD), "soul"), soul);
    Assert.assertEquals(RecordFiles.read(MajSoulProtoTest.response(true), "soul"), soul);
    Assert.assertEquals(RecordFiles.read(gzip(MajSoulProtoTest.response(true)), "soul"), soul);
  }

  /**
   * 次のものは読まずに IllegalArgumentException を投げる。空のバイト列、16 MiB を超えるバイト列（gzip なら展開した中身も）、壊れた
   * gzip、空白だけの中身、入れ子が 64 段を超える JSON、解析結果の JSON、ほかの形（base64、records だけの雀魂の配列）。形式のリーダーが
   * 投げた例外も、ほかの型なら IllegalArgumentException にして投げる。
   */
  @Test
  public void malformedRecordsAreRejected() {
    rejected(new byte[0], "between");
    rejected(new byte[RecordFiles.MAX_BYTES + 1], "between");
    rejected(gzip(new byte[RecordFiles.MAX_BYTES + 1]), "unpacks");
    rejected(new byte[] {0x1f, (byte) 0x8b, 1, 2, 3}, "gzip");
    rejected(utf8(" \n "), "empty");
    rejected(utf8("[".repeat(RecordFiles.MAX_NESTING + 1) + "]".repeat(RecordFiles.MAX_NESTING + 1)), "deeper");
    rejected(utf8("[".repeat(RecordFiles.MAX_NESTING) + "]".repeat(RecordFiles.MAX_NESTING)), "event objects");
    rejected(utf8("{\"format\":6,\"resultId\":\"x\"}"), "analysis");
    rejected(utf8("CgYKBAgBEAE="), "neither");
    rejected(utf8("[{\"name\":\".lq.RecordNewRound\",\"data\":{}}]"), "neither");
    rejected(utf8("{\"type\":\"start_game\",\"names\":\"A\"}"), "MJAI");
  }

  /** 属性を 50,000 個持つタグのある天鳳の XML も、そのタグが無いときと同じ半荘に読む。 */
  @Test
  public void aTagWithManyAttributesIsRead() {
    String id = "self-0";
    String xml = TenhouXml.write(RecordedGames.play(VARIANT, 0, id));
    StringBuilder attributes = new StringBuilder();
    for (int index = 0; index < 50_000; index++) {
      attributes.append(" a").append(index).append("=\"").append(index).append('"');
    }
    String wide = xml.replaceFirst("<mjloggm", "<mjloggm" + attributes);
    Assert.assertNotEquals(wide, xml);
    Assert.assertEquals(RecordFiles.read(utf8(wide), id), TenhouXml.read(xml, id));
  }

  /** 局のイベントが合わせて 50,000 を超える半荘は読まない。上限を超えない範囲なら読む。 */
  @Test
  public void aRecordWithTooManyEventsIsRejected() {
    RecordedGame game = RecordedGames.play(VARIANT, 1, "long");
    List<String> lines = Arrays.asList(MjaiCodec.write(game).split("\n"));
    int end = lines.indexOf("{\"type\":\"end_kyoku\"}");
    List<String> round = lines.subList(1, end + 1);
    int events = MjaiCodec.read(MjaiCodec.write(game), "long").rounds().get(0).events().size();
    byte[] within = repeated(lines, round, RecordFiles.MAX_EVENTS / events);
    Assert.assertEquals(RecordFiles.read(within, "within").rounds().size(), RecordFiles.MAX_EVENTS / events);
    rejected(repeated(lines, round, RecordFiles.MAX_EVENTS / events + 1), "50000");
  }

  /**
   * start_game の行、1 局の行を指定の回数、end_game の行の順に並べた mjai の牌譜を作る。
   *
   * @param lines mjai の牌譜の行（先頭が start_game、最後が end_game）
   * @param round 1 局の行
   * @param copies 局を並べる回数
   * @return UTF-8 のバイト列
   */
  private static byte[] repeated(List<String> lines, List<String> round, int copies) {
    StringBuilder text = new StringBuilder(lines.get(0)).append('\n');
    for (int copy = 0; copy < copies; copy++) {
      for (String line : round) {
        text.append(line).append('\n');
      }
    }
    text.append(lines.get(lines.size() - 1)).append('\n');
    byte[] bytes = utf8(text);
    Assert.assertTrue(bytes.length <= RecordFiles.MAX_BYTES, bytes.length + " bytes");
    return bytes;
  }

  private static void rejected(byte[] bytes, String reason) {
    IllegalArgumentException failure =
        Assert.expectThrows(IllegalArgumentException.class, () -> RecordFiles.read(bytes, "bad"));
    Assert.assertTrue(failure.getMessage().contains(reason), failure.getMessage());
  }

  private static byte[] utf8(CharSequence text) {
    return text.toString().getBytes(StandardCharsets.UTF_8);
  }

  private static byte[] gzip(byte[] bytes) {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (GZIPOutputStream gzip = new GZIPOutputStream(out)) {
      gzip.write(bytes);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return out.toByteArray();
  }
}
