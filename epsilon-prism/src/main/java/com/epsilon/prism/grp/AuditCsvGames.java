package com.epsilon.prism.grp;

import com.epsilon.prism.workflow.AuditGrpWorkflow;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * audit-grp の CSV（{@link AuditGrpWorkflow} の列）を半荘の記録に組み直したもの。CSV の kyoku・honba・sticks は終わった局の値なので、次局の
 * 整数を組み直す: 次局の供託 = (100000 − Σ s_i)/1000、次局の局と本場は同じ半荘の次の行から、半荘の最後の行（ended）には次局がない。
 *
 * @param games 半荘（CSV の game の順）
 * @param marginals 半荘が続く局末ごとの CSV の GRP の順位分布 p0..p15（{@link HandEndPhi#gameValues} と同じ順）
 * @param phi 半荘が続く局末ごとの CSV の Φ phi0..phi3（同じ順）
 */
public record AuditCsvGames(List<RecordedGame> games, float[][] marginals, double[][] phi) {
  /** 4 席の持ち点と供託の合計、供託 1 本の点（次局の供託 = (TOTAL − Σ s_i)/STICK）。 */
  private static final int TOTAL = 100_000, STICK = 1000;

  /** 読む列（CSV の見出しの名前で引く。p・phi・rank・s は先頭の列から 16・4・4・4 個続く）。 */
  private enum Column {
    GAME("game"),
    HAND("hand"),
    KYOKU("kyoku"),
    HONBA("honba"),
    SCORE("s0"),
    ENDED("ended"),
    MARGINAL("p0"),
    PHI("phi0"),
    RANK("rank0");

    private final String header;

    Column(String header) {
      this.header = header;
    }
  }

  /**
   * CSV（gzip）を読む。
   *
   * @throws IOException 読めないか、列・行の並びが audit-grp の形でないとき
   */
  public static AuditCsvGames read(Path csvGz) throws IOException {
    try (var reader =
        new BufferedReader(
            new InputStreamReader(
                new GZIPInputStream(Files.newInputStream(csvGz), 1 << 16), StandardCharsets.UTF_8),
            1 << 20)) {
      String header = reader.readLine();
      if (header == null) throw new IOException("Empty audit CSV " + csvGz);
      int[] at = columns(header.split(","));
      List<RecordedGame> games = new ArrayList<>();
      List<float[]> marginals = new ArrayList<>();
      List<double[]> phi = new ArrayList<>();
      List<String[]> rows = new ArrayList<>();
      long line = 1;
      for (String text = reader.readLine(); text != null; text = reader.readLine()) {
        line++;
        String[] row = text.split(",", -1);
        if (!rows.isEmpty() && !rows.getFirst()[at[Column.GAME.ordinal()]].equals(row[at[Column.GAME.ordinal()]])) {
          games.add(game(rows, at, marginals, phi, line));
          rows.clear();
        }
        rows.add(row);
      }
      if (!rows.isEmpty()) games.add(game(rows, at, marginals, phi, line));
      return new AuditCsvGames(
          List.copyOf(games), marginals.toArray(float[][]::new), phi.toArray(double[][]::new));
    }
  }

  /** 見出しから列の位置（{@link Column} の順）。 */
  private static int[] columns(String[] header) throws IOException {
    List<String> names = Arrays.asList(header);
    int[] at = new int[Column.values().length];
    for (Column column : Column.values()) {
      at[column.ordinal()] = names.indexOf(column.header);
      if (at[column.ordinal()] < 0) throw new IOException("Audit CSV lacks column " + column.header);
    }
    return at;
  }

  /** 1 半荘の行（局末の順）を記録に組み直し、半荘が続く局末の p と Φ を足す。 */
  private static RecordedGame game(
      List<String[]> rows, int[] at, List<float[]> marginals, List<double[]> phi, long line)
      throws IOException {
    String[] last = rows.getLast();
    long gameId = Long.parseLong(last[at[Column.GAME.ordinal()]]);
    List<HandStart> starts = new ArrayList<>(rows.size() - 1);
    for (int k = 0; k < rows.size(); k++) {
      String[] row = rows.get(k);
      boolean ended = k == rows.size() - 1;
      if (Integer.parseInt(row[at[Column.HAND.ordinal()]]) != k
          || Integer.parseInt(row[at[Column.ENDED.ordinal()]]) != (ended ? 1 : 0))
        throw new IOException("Audit CSV game " + gameId + " is out of order before line " + line);
      if (ended) break;
      int[] scores = ints(row, at[Column.SCORE.ordinal()]);
      int deposit = TOTAL - (scores[0] + scores[1] + scores[2] + scores[3]);
      if (deposit < 0 || deposit % STICK != 0)
        throw new IOException(
            "Audit CSV game " + gameId + " hand " + k + " has scores summing to " + (TOTAL - deposit));
      String[] next = rows.get(k + 1);
      starts.add(
          new HandStart(
              Integer.parseInt(next[at[Column.KYOKU.ordinal()]]),
              Integer.parseInt(next[at[Column.HONBA.ordinal()]]),
              deposit / STICK,
              scores));
      float[] p = new float[16];
      for (int i = 0; i < p.length; i++) p[i] = Float.parseFloat(row[at[Column.MARGINAL.ordinal()] + i]);
      marginals.add(p);
      double[] values = new double[4];
      for (int seat = 0; seat < 4; seat++)
        values[seat] = Double.parseDouble(row[at[Column.PHI.ordinal()] + seat]);
      phi.add(values);
    }
    RecordedGame game = new RecordedGame(gameId, starts, ints(last, at[Column.SCORE.ordinal()]));
    if (!Arrays.equals(game.finalRanks(), ints(last, at[Column.RANK.ordinal()])))
      throw new IOException("Audit CSV game " + gameId + " final ranks disagree with its final scores");
    return game;
  }

  private static int[] ints(String[] row, int from) {
    int[] values = new int[4];
    for (int i = 0; i < 4; i++) values[i] = Integer.parseInt(row[from + i]);
    return values;
  }
}
