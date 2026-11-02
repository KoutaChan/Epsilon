package com.epsilon.prism.grp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.Strictness;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * reports/phimon.jsonl の 1 行: コホート 1 つを使用中の Φ と影の Φ で採点した結果（{@link PhiScorer}）。EWMA（λ = 1/8）は使用中の項目が
 * 変わるごとに初めからやり直す。
 *
 * @param cohort コホート t
 * @param phiId 使用中の項目の id（コホートの記録の先頭のもの）
 * @param phiSha256 その sha
 * @param jarSha256 作業が動いた jar の sha
 * @param active 使用中の Φ の測り
 * @param shadows 影の Φ の測り（項目か試行の id → 測り。元の GRP は {@value PhiEntry#FIXED_ID}）
 * @param dSpgEwma 使用中の D_SPG の EWMA
 * @param ewmaCohorts EWMA に入ったコホートの数（項目が変わってから）
 * @param seconds 採点の秒数
 */
public record PhiMonitorLine(
    int cohort,
    String phiId,
    String phiSha256,
    String jarSha256,
    CellReport active,
    Map<String, CellReport> shadows,
    double dSpgEwma,
    int ewmaCohorts,
    double seconds) {

  private static final Logger LOG = LoggerFactory.getLogger(PhiMonitorLine.class);

  /** EWMA の重み λ。 */
  public static final double EWMA_WEIGHT = 1.0 / 8;

  /** 1 行 1 物（NaN の測りもそのまま書いて読む）。 */
  private static final Gson LINES =
      new GsonBuilder().serializeSpecialFloatingPointValues().setStrictness(Strictness.LENIENT).create();

  /**
   * phimon.jsonl の全ての行（書いた順。ファイルがなければ空）。壊れた行は、最後の行でなくても書きかけの残り（次の {@link #append} が改行で
   * 閉じる）として読み飛ばす。
   */
  public static List<PhiMonitorLine> readAll(GrpFiles files) throws IOException {
    Path path = files.monitor();
    List<PhiMonitorLine> lines = new ArrayList<>();
    if (!Files.exists(path)) return lines;
    List<String> texts = Files.readAllLines(path, StandardCharsets.UTF_8);
    for (int i = 0; i < texts.size(); i++) {
      String text = texts.get(i);
      if (text.isBlank()) continue;
      try {
        lines.add(parse(text));
      } catch (JsonParseException | IllegalStateException | NullPointerException broken) {
        LOG.warn("skipped the broken line {} of {}", i + 1, path);
      }
    }
    return lines;
  }

  /** phimon.jsonl に 1 行足す。 */
  public void append(GrpFiles files) throws IOException {
    Path path = files.monitor();
    Files.createDirectories(path.getParent());
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
      // 書きかけで止まった最後の行があれば改行で閉じてから足す（読み手はその行を飛ばす）
      long size = channel.size();
      String prefix = "";
      if (size > 0) {
        ByteBuffer lastByte = ByteBuffer.allocate(1);
        channel.read(lastByte, size - 1);
        if (lastByte.get(0) != '\n') prefix = "\n";
      }
      channel.write(ByteBuffer.wrap((prefix + LINES.toJson(this) + "\n").getBytes(StandardCharsets.UTF_8)), size);
      channel.force(true);
    }
  }

  /**
   * 走らせ役の要約の行（phi[t] D=… B4=… S3=… z=…）。D は D_SPG とその EWMA（コホートの数）、B4 は今 4 位の席の残差の E2〜S2 の平均、S3 は
   * 南 3 の同じ量、z は傾きの z の絶対値が最大の区分とその z。
   */
  public String summary() {
    double b4 = 0;
    int groups = 0;
    for (HandGroup group : List.of(HandGroup.E2, HandGroup.E3, HandGroup.E4, HandGroup.S1, HandGroup.S2)) {
      double value = active.b4()[group.ordinal()];
      if (Double.isNaN(value)) continue;
      b4 += value;
      groups++;
    }
    HandGroup widest = null;
    for (HandGroup group : HandGroup.values()) {
      double z = active.slopeZ()[group.ordinal()];
      if (!Double.isNaN(z) && (widest == null || Math.abs(z) > Math.abs(active.slopeZ()[widest.ordinal()])))
        widest = group;
    }
    return String.format(
        Locale.ROOT,
        "phi[%d] %s D=%.3e ewma=%.3e(%d) B4=%+.4f S3=%+.4f z=%s mse=%.5f shadows=%d %.0fs",
        cohort,
        phiId,
        active.dSpg(),
        dSpgEwma,
        ewmaCohorts,
        groups == 0 ? Double.NaN : b4 / groups,
        active.b4()[HandGroup.S3.ordinal()],
        widest == null ? "nan" : String.format(Locale.ROOT, "%s:%+.1f", widest, active.slopeZ()[widest.ordinal()]),
        active.mse(),
        shadows.size(),
        seconds);
  }

  /** 1 行を読む（欄が欠けた行は拒む）。 */
  static PhiMonitorLine parse(String text) {
    PhiMonitorLine line = LINES.fromJson(text, PhiMonitorLine.class);
    if (line == null || line.phiId() == null || line.active() == null || line.shadows() == null)
      throw new JsonParseException("Incomplete phimon line");
    return line;
  }

}
