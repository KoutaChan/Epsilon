package com.epsilon.prism.learn;

import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.learn.DataParallelGroup;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.FitProbe;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.settings.DiagnosticProperty;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.function.ToLongFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の計測。システムプロパティ {@value #PROPERTY}={@code sync:K:W} で有効にする。既定は無効で、学習の処理は何も変えない。
 * ウォームアップの W ステップの後のステップを 0 から数え、番号が K の倍数のステップを同期するステップに、残りをホストだけのステップにする
 * （{@link Timing}）。
 *
 * <ul>
 *   <li>同期するステップ: 段の区切りの印（{@link FitProbe#mark}）ごとにレーンの計算ストリームを待つ。だから段の時間は、その段だけの
 *       「ホストが処理を積む時間とデバイスの計算時間の長い方」になる。ステップの始め、レーンの処理の後、最適化の後に全てのレーンを待つ
 *       （{@link DataParallelGroup#awaitParameterUpdates}）ので、最適化の区間も最適化だけの時間になる。
 *   <li>ホストだけのステップ: 印ではタイマーを読むだけ（本番と同じ流れ）。ホストが待つ所（アンカー・配列の作成・損失の回収・最適化の検査）に
 *       前の処理の残りが入るので、同期するステップとの差が、そこで吸収した待ち時間になる。
 * </ul>
 *
 * <p>計測するときは、ヘッドの中と埋め込みを計測するための葉も作る（{@link FitProbe#cuts}。ウォームアップのステップを含め、どのステップでも
 * 同じ木）。そして逆伝播を、ヘッドの中と埋め込みの段に分ける（バックボーンの出力は Attention の記録の正規化の葉なので、ヘッドの側と
 * バックボーンの側にはいつも分かれる）。学習の終わりに、段ごとのステップあたりの ms、ステップの区間（レーンの処理・最適化）、レーンの
 * マイクロバッチの形（ステップあたり）の表をログに出す（{@link #log}）。段ごとの ms は、レーンごとと「遅いレーン」（ステップごとに忙しい
 * 時間が長い方のレーン）について、同期するステップとホストだけのステップに分けて出す。
 */
final class FitProfile {

  /** 計測を選ぶシステムプロパティ。 */
  static final String PROPERTY = DiagnosticProperty.FIT_TIMING.property();

  /** システムプロパティの値の形（sync:K:W）。 */
  private static final Pattern SETTING = Pattern.compile("sync:(\\d{1,6}):(\\d{1,6})");

  /** 段と層の組の数（層ごとの段は、バックボーンの層の上限まで数える）。 */
  private static final int SLOTS = Stage.values().length * PrismShape.MAX_DEPTH;

  /** 表の各行の先頭に付ける文字列。 */
  private static final String PREFIX = "fit profile ";

  private static final Logger LOG = LoggerFactory.getLogger(FitProfile.class);

  /** ステップの計り方。 */
  enum Timing {
    /** 計らない（計測しない学習と、ウォームアップのステップ）。 */
    OFF,
    /** 印ではタイマーを読むだけ（本番と同じ流れ）。 */
    HOST,
    /** 印ごとにレーンの計算ストリームを待つ。 */
    SYNCED;

    /**
     * 同期するステップなら、全てのレーンに積んだ処理が終わるのを待つ。
     *
     * @param group 学習のレーン
     */
    void settle(DataParallelGroup<?> group) {
      if (this == SYNCED) {
        group.awaitParameterUpdates();
      }
    }
  }

  /** レーンのマイクロバッチの形（ステップごとの値を足す）。 */
  private enum Measure {
    /** マイクロバッチの数。 */
    PIECES,
    /** 行の数。 */
    ROWS,
    /** いちばん長い行の S（レーンのマイクロバッチの中で最大）。 */
    WIDEST_S,
    /** パディングを含むトークンの数（Σ 行 × S）。 */
    PADDED_TOKENS,
    /** パディングを含むトークンの組の数（Σ 行 × S²。Attention の計算量の目安）。 */
    PADDED_PAIRS
  }

  /** 1 つの計り方について、ステップの時間の和（ns）。 */
  private static final class Tally {

    private int steps;
    private long compute;
    private long optimizer;

    /** [レーン][段と層の組]。 */
    private final long[][] lanes;

    /** レーンの忙しい時間（処理の始めから最後の印まで）。 */
    private final long[] busy;

    /** ステップごとの遅いレーンの、[段と層の組] ごとの時間と忙しい時間。 */
    private final long[] slowest = new long[SLOTS];

    private long slowestBusy;

    private Tally(int lanes) {
      this.lanes = new long[lanes][SLOTS];
      busy = new long[lanes];
    }
  }

  /** 設定の文字列（無効なら null）。 */
  private final String setting;

  private final int period;
  private final int warmup;
  private final List<Clock> clocks = new ArrayList<>();
  private final Map<Timing, Tally> tallies = new EnumMap<>(Timing.class);

  /** [レーン][形]（計るステップの和。最初に計るステップで作る）。 */
  private long[][] shapes;

  private int measured;

  private FitProfile(String setting, int period, int warmup) {
    this.setting = setting;
    this.period = period;
    this.warmup = warmup;
  }

  /**
   * システムプロパティ {@value #PROPERTY} から作る。
   *
   * @return 計測（プロパティがなければ無効）
   */
  static FitProfile configured() {
    return of(System.getProperty(PROPERTY));
  }

  /**
   * 設定から作る。
   *
   * @param setting sync:K:W（K ステップに 1 回同期する。K ≥ 1。ウォームアップは W ステップ）。null か空なら無効
   * @return 計測
   * @throws IllegalArgumentException 形が違うとき
   */
  static FitProfile of(String setting) {
    if (setting == null || setting.isBlank()) {
      return new FitProfile(null, 0, 0);
    }
    Matcher matcher = SETTING.matcher(setting.strip());
    if (!matcher.matches() || Integer.parseInt(matcher.group(1)) < 1) {
      throw new IllegalArgumentException(
          PROPERTY + " must be sync:K:W with K >= 1 and W >= 0, got " + setting);
    }
    return new FitProfile(
        setting.strip(), Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
  }

  boolean enabled() {
    return setting != null;
  }

  /**
   * ステップの計り方を返す。
   *
   * @param step ステップの番号（1 から）
   * @return 計り方（無効か、ウォームアップのステップなら {@link Timing#OFF}）
   */
  Timing timing(int step) {
    if (!enabled() || step <= warmup) {
      return Timing.OFF;
    }
    return (step - warmup - 1) % period == 0 ? Timing.SYNCED : Timing.HOST;
  }

  /**
   * レーンのタイマーを作る。作った順がレーンの番号になる。
   *
   * @param stream レーンの計算ストリーム（CPU のレーンは null）
   * @return タイマー（閉じるのはレーン）
   */
  Clock clock(PtStream stream) {
    Clock clock = new Clock(enabled() && stream != null ? stream.newEvent() : null);
    clocks.add(clock);
    return clock;
  }

  /**
   * 計るステップの時間とマイクロバッチの形を足す。ほかのステップでは何もしない。全てのレーンの処理が返った後に、学習のスレッドで呼ぶ。
   *
   * @param timing ステップの計り方
   * @param step ステップのマイクロバッチ
   * @param compute レーンの処理の区間（ns）
   * @param optimizer 最適化の区間（ns）
   */
  void record(Timing timing, StepFeed.Step step, long compute, long optimizer) {
    if (timing == Timing.OFF) {
      return;
    }
    Tally tally = tallies.computeIfAbsent(timing, ignored -> new Tally(clocks.size()));
    tally.steps++;
    tally.compute += compute;
    tally.optimizer += optimizer;
    Clock slowest = null;
    for (int lane = 0; lane < clocks.size(); lane++) {
      Clock clock = clocks.get(lane);
      // マイクロバッチのないレーンは、このステップの処理を始めていない
      if (clock.timing != Timing.OFF) {
        tally.busy[lane] += clock.busy();
        add(tally.lanes[lane], clock.nanos);
        if (slowest == null || clock.busy() > slowest.busy()) {
          slowest = clock;
        }
      }
    }
    if (slowest != null) {
      tally.slowestBusy += slowest.busy();
      add(tally.slowest, slowest.nanos);
    }
    for (Clock clock : clocks) {
      clock.timing = Timing.OFF;
    }
    if (shapes == null) {
      shapes = new long[clocks.size()][Measure.values().length];
    }
    for (int lane = 0; lane < clocks.size(); lane++) {
      addShapes(shapes[lane], step.lanes().get(lane));
    }
    measured++;
  }

  /** 表をログに出す。計るステップがなければ何もしない。 */
  void log() {
    for (String line : report()) {
      LOG.info(line);
    }
  }

  /**
   * 表の行を返す。順に、見出し、ステップの区間（同期するステップ・ホストだけのステップ）、レーンのマイクロバッチの形、段ごとの ms
   * （どれかの列が 0 でない段と層だけ）。数は全て、その計り方のステップあたり。
   *
   * @return 行（計るステップがなければ空）
   */
  List<String> report() {
    List<String> lines = new ArrayList<>();
    if (measured == 0) {
      return lines;
    }
    Tally synced = tallies.get(Timing.SYNCED);
    Tally host = tallies.get(Timing.HOST);
    int lanes = clocks.size();
    lines.add(
        PREFIX
            + setting
            + ": lanes "
            + lanes
            + ", synced steps "
            + steps(synced)
            + ", host-only steps "
            + steps(host)
            + " (ms per step; slow = the lane with the longer busy time in each step)");
    lines.add(header("window", "synced", "host"));
    lines.add(row("compute", millis(synced, t -> t.compute), millis(host, t -> t.compute)));
    lines.add(row("optimizer", millis(synced, t -> t.optimizer), millis(host, t -> t.optimizer)));
    String[] laneNames = new String[lanes];
    for (int lane = 0; lane < lanes; lane++) {
      laneNames[lane] = "lane" + lane;
    }
    lines.add(header("shape", laneNames));
    for (Measure measure : Measure.values()) {
      double[] values = new double[lanes];
      for (int lane = 0; lane < lanes; lane++) {
        values[lane] = shapes[lane][measure.ordinal()] / (double) measured;
      }
      lines.add(row(measure.name(), values));
    }
    String[] columns = new String[2 * lanes + 2];
    for (int lane = 0; lane < lanes; lane++) {
      columns[2 * lane] = "lane" + lane + " sync";
      columns[2 * lane + 1] = "lane" + lane + " host";
    }
    columns[2 * lanes] = "slow sync";
    columns[2 * lanes + 1] = "slow host";
    lines.add(header("stage", columns));
    for (Stage stage : Stage.values()) {
      for (int layer = 0; layer < (stage.layered() ? PrismShape.MAX_DEPTH : 1); layer++) {
        int slot = slot(stage, layer);
        double[] values = new double[columns.length];
        for (int lane = 0; lane < lanes; lane++) {
          final int index = lane;
          values[2 * lane] = millis(synced, t -> t.lanes[index][slot]);
          values[2 * lane + 1] = millis(host, t -> t.lanes[index][slot]);
        }
        values[2 * lanes] = millis(synced, t -> t.slowest[slot]);
        values[2 * lanes + 1] = millis(host, t -> t.slowest[slot]);
        if (Arrays.stream(values).anyMatch(value -> value > 0)) {
          lines.add(row(stage.layered() ? stage + " L" + layer : stage.name(), values));
        }
      }
    }
    double[] busy = new double[columns.length];
    for (int lane = 0; lane < lanes; lane++) {
      final int index = lane;
      busy[2 * lane] = millis(synced, t -> t.busy[index]);
      busy[2 * lane + 1] = millis(host, t -> t.busy[index]);
    }
    busy[2 * lanes] = millis(synced, t -> t.slowestBusy);
    busy[2 * lanes + 1] = millis(host, t -> t.slowestBusy);
    lines.add(row("BUSY", busy));
    return lines;
  }

  /** 段と層の組の番号。 */
  private static int slot(Stage stage, int layer) {
    if (layer < 0 || layer >= (stage.layered() ? PrismShape.MAX_DEPTH : 1)) {
      throw new IllegalArgumentException("invalid layer " + layer + " for the stage " + stage);
    }
    return stage.ordinal() * PrismShape.MAX_DEPTH + layer;
  }

  /** レーンの 1 ステップ分のマイクロバッチの形を足す（S はパディングを含むトークン数 / 行数）。 */
  private static void addShapes(long[] sums, List<Microbatch> parts) {
    long widest = 0;
    for (Microbatch part : parts) {
      long width = part.tokens() / part.rows();
      sums[Measure.PIECES.ordinal()]++;
      sums[Measure.ROWS.ordinal()] += part.rows();
      sums[Measure.PADDED_TOKENS.ordinal()] += part.tokens();
      sums[Measure.PADDED_PAIRS.ordinal()] += part.tokens() * width;
      widest = Math.max(widest, width);
    }
    sums[Measure.WIDEST_S.ordinal()] += widest;
  }

  private static void add(long[] sums, long[] values) {
    for (int i = 0; i < sums.length; i++) {
      sums[i] += values[i];
    }
  }

  private static int steps(Tally tally) {
    return tally == null ? 0 : tally.steps;
  }

  /** その計り方のステップあたりの ms（その計り方のステップがなければ NaN）。 */
  private static double millis(Tally tally, ToLongFunction<Tally> nanos) {
    return tally == null ? Double.NaN : nanos.applyAsLong(tally) / 1e6 / tally.steps;
  }

  private static String header(String label, String... columns) {
    StringBuilder line = new StringBuilder(PREFIX).append(String.format("%-22s", label));
    for (String column : columns) {
      line.append(String.format("%12s", column));
    }
    return line.toString();
  }

  private static String row(String label, double... values) {
    StringBuilder line = new StringBuilder(PREFIX).append(String.format("%-22s", label));
    for (double value : values) {
      line.append(String.format("%12.2f", value));
    }
    return line.toString();
  }

  /**
   * 1 つのレーンの段ごとのタイマー（{@link FitProbe}）。印を付けるのはレーンのスレッドだけで、ステップの終わりに学習のスレッドが読む。
   * 学習のスレッドはレーンの処理が返るのを待ってから読むので、その待ちが読み書きの順序を保証する。
   */
  final class Clock implements FitProbe, AutoCloseable {

    /** 同期するステップで、レーンの計算ストリームに置いて待つイベント（GPU のレーンを計測するときだけ。ほかは null）。 */
    private final PtEvent event;

    /** 今のステップの、段と層の組ごとの時間（ns）。 */
    private final long[] nanos = new long[SLOTS];

    private Timing timing = Timing.OFF;
    private long begun;
    private long last;

    private Clock(PtEvent event) {
      this.event = event;
    }

    /**
     * このステップでのレーンの処理を始める。同期するステップでは、前の処理が終わるのを待ってからタイマーを始める。
     *
     * @param step ステップの計り方
     */
    void begin(Timing step) {
      timing = step;
      if (timing != Timing.OFF) {
        settle();
        Arrays.fill(nanos, 0);
        begun = System.nanoTime();
        last = begun;
      }
    }

    @Override
    public void mark(Stage stage, int layer) {
      if (timing == Timing.OFF) {
        return;
      }
      settle();
      long now = System.nanoTime();
      nanos[slot(stage, layer)] += now - last;
      last = now;
    }

    @Override
    public boolean cuts() {
      return enabled();
    }

    /** ステップの処理の始めから最後の印までの時間（ns）。 */
    private long busy() {
      return last - begun;
    }

    /** 同期するステップなら、レーンの計算ストリームに積んだ処理が終わるのを待つ。 */
    private void settle() {
      if (timing == Timing.SYNCED && event != null) {
        event.record();
        event.synchronize();
      }
    }

    @Override
    public void close() {
      if (event != null) {
        event.close();
      }
    }
  }
}
