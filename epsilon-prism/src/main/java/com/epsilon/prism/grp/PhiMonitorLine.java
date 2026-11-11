package com.epsilon.prism.grp;

import com.epsilon.grp.train.GrpReport;
import com.epsilon.grp.train.LateReport;
import com.epsilon.grp.train.OutcomeCalibration;
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
 * reports/phimon.jsonl の 1 行。コホート 1 つを {@link PhiScorer} で採点した結果を持つ。採点には、使ったモードの値、同じ GRP の項目の
 * もう一方のモードの値、影の Φ を使う。もう一方のモードの採点・終盤の報告・q の較正は、項目が q を持つときだけ入る。
 *
 * @param cohort コホート t
 * @param phiId 使った GRP の項目の id（コホートの記録の先頭に書いたもの）
 * @param mode 使ったモード
 * @param cutoff LATE の打ち切りの値。GRP のモードでは null
 * @param active 使ったモードの値で採点した結果
 * @param other 同じ項目のもう一方のモードの値で採点した結果。項目が q を持たないか、もう一方のモードの値を開けなければ null
 * @param late LATE が数え上げる局末で、LATE の値と GRP のモードの値を比べた報告。もう一方のモードの採点が無ければ null
 * @param outcomeCalibration 数のファイルの局のうち、LATE が数え上げる局で測った項目の q の較正。項目が q を持たないか、数のファイルが
 *     無ければ null
 * @param shadows 影の Φ ごとに、使ったモードの値で採点した結果。キーは項目か試行の id で、元の GRP は {@value PhiEntry#FIXED_ID}
 * @param seconds 採点にかかった秒数
 */
public record PhiMonitorLine(
    int cohort,
    String phiId,
    PhiMode mode,
    Double cutoff,
    GrpReport active,
    GrpReport other,
    LateReport late,
    OutcomeCalibration.Check outcomeCalibration,
    Map<String, GrpReport> shadows,
    double seconds) {

  private static final Logger LOG = LoggerFactory.getLogger(PhiMonitorLine.class);

  /** 1 行に JSON を 1 つ書く Gson。NaN の値もそのまま書き、そのまま読む。 */
  private static final Gson LINES =
      new GsonBuilder().serializeSpecialFloatingPointValues().setStrictness(Strictness.LENIENT).create();

  /**
   * phimon.jsonl の全ての行を書いた順に返す。ファイルが無ければ空。壊れた行は、最後の行でなくても書きかけで止まった行とみなして
   * 読み飛ばす（そうした行は次の {@link #append} が改行で閉じる）。モード・打ち切りの値・もう一方のモードの採点・終盤の報告・q の較正の
   * フィールドが無い行も読み、無いフィールドは null になる。
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
      // 最後の行が書きかけで止まっていれば、改行で閉じてから足す。読む側はその行を飛ばす
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
   * ログに書く 1 行の要約を返す。並ぶのは順に、phi[t]、id、mse、D、終盤の 1 つ手前の区分の傾きとその z、僅差のセルの
   * |実測 − 予測| の最大（ポイント）、使ったモード、LATE が数え上げる局末での対の MSE の差（LATE − GRP のモード）とその z、影の数、
   * 秒数。
   */
  public String summary() {
    List<GrpReport.Fit> groups = active.groups();
    GrpReport.Fit penultimate = groups.size() >= 2 ? groups.get(groups.size() - 2) : null;
    return String.format(
        Locale.ROOT,
        "phi[%d] %s mse=%.5f D=%.3e slope=%.3f z=%+.1f close=%.2fpp mode=%s late=%s shadows=%d %.0fs",
        cohort,
        phiId,
        active.mse(),
        active.dSpg(),
        penultimate == null ? Double.NaN : penultimate.slope(),
        penultimate == null ? Double.NaN : penultimate.slopeZ(),
        100 * widestCloseGap(active),
        mode,
        late == null ? "-" : String.format(Locale.ROOT, "%+.3e(z=%+.1f)", late.mse().mean(), late.mse().z()),
        shadows.size(),
        seconds);
  }

  /** 僅差のセルの |実測 − 予測| の最大。測れないセルは除き、どのセルも測れなければ NaN。 */
  private static double widestCloseGap(GrpReport report) {
    double widest = Double.NaN;
    for (GrpReport.CloseCell cell : report.closeCells()) {
      double gap = Math.abs(cell.keepGap());
      if (Double.isFinite(gap) && !(gap <= widest)) widest = gap;
    }
    return widest;
  }

  /** 1 行を読む。項目の id・使ったモードの採点・影の採点のどれかが欠けた行は拒む。 */
  static PhiMonitorLine parse(String text) {
    PhiMonitorLine line = LINES.fromJson(text, PhiMonitorLine.class);
    if (line == null || line.phiId() == null || line.active() == null || line.shadows() == null)
      throw new JsonParseException("Incomplete phimon line");
    return line;
  }
}
