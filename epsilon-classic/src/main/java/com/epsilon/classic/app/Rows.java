package com.epsilon.classic.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.data.ClassicRows;
import com.epsilon.classic.data.LogRows;
import com.epsilon.classic.data.RecordDirectory;
import com.epsilon.classic.data.RowChunks;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 牌譜のディレクトリを再生して、学習の行のディレクトリを書く（{@link LogRows}）。牌譜のファイルを名前の順に区切り、区切りごとに行の
 * ファイルを 1 つ書く（{@link RowChunks}）。区切りは牌譜のファイルの数で決めるので、飛ばした牌譜があっても、同じ牌譜はいつも同じ
 * ファイルに入る。読めない牌譜と、記録がエンジンと食い違う牌譜は飛ばして数え、最初の 1 つだけをログに書く。
 *
 * @param out 書く行のディレクトリ。既にあれば何も書かず、使い方の誤りとして扱う
 * @param chunkGames 1 つの行のファイルに入れる牌譜の数。行のファイルの判断の記録（O）は 1 つの配列にまとめて置くので、配列の上限に
 *     収まる数にする
 */
@Named(value = "rows", doc = "Replay a records directory and write a training rows directory")
public record Rows(
    @Key(doc = "Records directory (files directly inside, read in name order)", presence = Presence.REQUIRED) Path records,
    @Key(doc = "Rows directory to write (must not exist yet)", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Rules", value = "TENHOU_4P") Variant variant,
    @Key(doc = "Fraction of games held out for validation (chosen by a hash of the record id)", value = "0.1")
        @Range(min = 0, minInclusive = false, max = 1, maxInclusive = false)
        double validation,
    @Key(doc = "Records per row file", value = "2048") @Range(min = 1) int chunkGames)
    implements ClassicCommand {

  private static final Logger LOG = LoggerFactory.getLogger(Rows.class);

  /** 書いた行の数、半荘の数とそのうち検証用の数、行のファイルの数、飛ばした牌譜の数。 */
  public record Result(Path out, long rows, int games, int validationGames, int chunks, int skipped) {}

  @Override
  public Outcome run(Context<ClassicSettings> context) throws IOException {
    Result result = write();
    return Outcome.ok(
        String.format(
            Locale.ROOT,
            "out=%s rows=%d games=%d validationGames=%d chunks=%d skipped=%d",
            result.out(), result.rows(), result.games(), result.validationGames(), result.chunks(), result.skipped()));
  }

  /**
   * 行のディレクトリを書く。
   *
   * @throws SettingsException 行のディレクトリが既にあるとき
   */
  public Result write() throws IOException {
    Inputs.requireNew(out, "rows directory");
    Inputs.requireDirectory(records, "records");
    List<Path> files = RecordDirectory.files(records);
    Files.createDirectories(out);
    LogRows logs = new LogRows(variant, validation);
    long rows = 0;
    int games = 0;
    int held = 0;
    int skipped = 0;
    int chunks = 0;
    for (int from = 0; from < files.size(); from += chunkGames) {
      ClassicRows.Builder builder = ClassicRows.builder(variant);
      for (Path file : files.subList(from, Math.min(files.size(), from + chunkGames))) {
        try {
          RecordedGame game = RecordDirectory.read(file);
          logs.add(game, builder);
        } catch (IllegalArgumentException | IllegalStateException unusable) {
          if (skipped++ == 0) {
            LOG.warn("rows skip {}: {}", file, unusable.toString());
          }
        }
      }
      ClassicRows chunk = builder.build();
      chunk.write(RowChunks.file(out, chunks++));
      rows += chunk.size();
      games += chunk.games().size();
      held += (int) chunk.games().stream().filter(ClassicRows.Game::validation).count();
      LOG.info(
          "rows chunk {}: games={} rows={} (total games={} rows={})",
          chunks - 1, chunk.games().size(), chunk.size(), games, rows);
    }
    return new Result(out, rows, games, held, chunks, skipped);
  }
}
