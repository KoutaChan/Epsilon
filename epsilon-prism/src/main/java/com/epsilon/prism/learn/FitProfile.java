package com.epsilon.prism.learn;

import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.model.FitProbe;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.training.DataParallelGroup;
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
 * 当てはめの計測（系の性質 {@value #PROPERTY}={@code sync:K:W}。既定は無効で、学習の道は何も変えない）。温めの W 歩の後の歩を 0 から
 * 数え、番号が K の倍数の歩を同期する歩、残りをホストだけの歩にする（{@link Timing}）。
 *
 * <ul>
 *   <li>同期する歩: レーンの段の印（{@link FitProbe#mark}）ごとにレーンの計算ストリームを待つので、段の時間はその段だけの「ホストの
 *       積み込みとデバイスの計算の長い方」になる。歩の始め・レーンの仕事の後・最適化の後に全てのレーンを待つ
 *       （{@link DataParallelGroup#awaitParameterUpdates}）ので、最適化の窓も最適化だけの時間になる。
 *   <li>ホストだけの歩: 印は時計を読むだけ（本番と同じ段取り）。ホストが待つ所（錨・配列の作成・損失の回収・最適化の検査）に前の仕事の
 *       残りが入るので、同期する歩との差がその所で吸った待ちになる。
 * </ul>
 *
 * <p>計測のときは背骨の出力も葉にして（{@link FitProbe#cuts}。温めの歩を含むどの歩でも同じ木）、逆伝播を頭の側と背骨の側に分ける。当てはめ
 * の終わりに、段ごとの歩あたりの ms（レーンごとと、歩ごとに忙しい時間の長い方のレーン「遅いレーン」の、同期する歩とホストだけの歩）、歩の
 * 窓（レーンの仕事・最適化）、レーンの微小バッチの形（歩あたり）の表を記録に出す（{@link #log}）。
 */
final class FitProfile {

  /** 計測を選ぶ系の性質。 */
  static final String PROPERTY = "prism.fitProfile";

  /** 系の性質の値の形（sync:K:W）。 */
  private static final Pattern SETTING = Pattern.compile("sync:(\\d{1,6}):(\\d{1,6})");

  /** 段と層の組の数（層の段は背骨の層の上限まで数える）。 */
  private static final int SLOTS = Stage.values().length * PrismShape.MAX_DEPTH;

  /** 表の行の頭。 */
  private static final String PREFIX = "fit profile ";

  private static final Logger LOG = LoggerFactory.getLogger(FitProfile.class);

  /** 歩の計り方。 */
  enum Timing {
    /** 計らない（計測しない当てはめと、温めの歩）。 */
    OFF,
    /** 印は時計を読むだけ（本番の段取り）。 */
    HOST,
    /** 印ごとにレーンの計算ストリームを待つ。 */
    SYNCED;

    /**
     * 同期する歩なら、全てのレーンの積んだ仕事の終わりを待つ。
     *
     * @param group 学習のレーン
     */
    void settle(DataParallelGroup<?> group) {
      if (this == SYNCED) {
        group.awaitParameterUpdates();
      }
    }
  }

  /** レーンの微小バッチの形（歩ごとの値を足す）。 */
  private enum Measure {
    /** 塊（微小バッチ）の数。 */
    PIECES,
    /** 行の数。 */
    ROWS,
    /** 最も長い行の S（レーンの塊で最大）。 */
    WIDEST_S,
    /** 詰めたトークン（Σ 行 × S）。 */
    PADDED_TOKENS,
    /** 詰めた組（Σ 行 × S²。注意の手間の目安）。 */
    PADDED_PAIRS
  }

  /** 1 つの計り方の歩の和（ns）。 */
  private static final class Tally {

    private int steps;
    private long compute;
    private long optimizer;

    /** [レーン][段と層の組]。 */
    private final long[][] lanes;

    /** レーンの忙しい時間（仕事の始めから最後の印まで）。 */
    private final long[] busy;

    /** 歩ごとの遅いレーンの [段と層の組] と忙しい時間。 */
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

  /** [レーン][形]（計る歩の和。最初の計る歩で作る）。 */
  private long[][] shapes;

  private int measured;

  private FitProfile(String setting, int period, int warmup) {
    this.setting = setting;
    this.period = period;
    this.warmup = warmup;
  }

  /**
   * 系の性質 {@value #PROPERTY} から作る。
   *
   * @return 計測（性質がなければ無効）
   */
  static FitProfile configured() {
    return of(System.getProperty(PROPERTY));
  }

  /**
   * 設定から作る。
   *
   * @param setting sync:K:W（K ≥ 1 歩に 1 歩を同期、温め W 歩）。null か空なら無効
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
   * 歩の計り方を返す。
   *
   * @param step 歩の番号（1 から）
   * @return 計り方（無効か温めの歩なら {@link Timing#OFF}）
   */
  Timing timing(int step) {
    if (!enabled() || step <= warmup) {
      return Timing.OFF;
    }
    return (step - warmup - 1) % period == 0 ? Timing.SYNCED : Timing.HOST;
  }

  /**
   * レーンの時計を作る（作る順がレーンの番号）。
   *
   * @param stream レーンの計算ストリーム（CPU のレーンは null）
   * @return 時計（閉じるのはレーン）
   */
  Clock clock(PtStream stream) {
    Clock clock = new Clock(enabled() && stream != null ? stream.newEvent() : null);
    clocks.add(clock);
    return clock;
  }

  /**
   * 計る歩の時間と微小バッチの形を足す（ほかの歩は何もしない）。全てのレーンの仕事が返った後に学習のスレッドで呼ぶ。
   *
   * @param timing 歩の計り方
   * @param step 歩の微小バッチ
   * @param compute レーンの仕事の窓（ns）
   * @param optimizer 最適化の窓（ns）
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
      // 微小バッチのないレーンはこの歩の仕事を始めていない
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

  /** 表を記録に出す（計る歩がなければ何もしない）。 */
  void log() {
    for (String line : report()) {
      LOG.info(line);
    }
  }

  /**
   * 表の行を返す: 見出し、歩の窓（同期する歩・ホストだけの歩）、レーンの微小バッチの形、段ごとの ms（どこかの列が 0 でない段と層だけ）。
   * 数は全て計り方の歩あたり。
   *
   * @return 行（計る歩がなければ空）
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

  /** レーンの歩の微小バッチの形を足す（S は詰めたトークン / 行）。 */
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

  /** 計り方の歩あたりの ms（その計り方の歩がなければ NaN）。 */
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
   * 1 レーンの段の時計（{@link FitProbe}）。印を付けるのはレーンのスレッドだけで、歩の終わりに学習のスレッドが読む（レーンの仕事が返るのを
   * 待ってから読むので、その待ちが読み書きの順を決める）。
   */
  final class Clock implements FitProbe, AutoCloseable {

    /** 同期する歩でレーンの計算ストリームに置いて待つ印（GPU のレーンを計測するときだけ。ほかは null）。 */
    private final PtEvent event;

    /** 今の歩の段と層の組ごとの時間（ns）。 */
    private final long[] nanos = new long[SLOTS];

    private Timing timing = Timing.OFF;
    private long begun;
    private long last;

    private Clock(PtEvent event) {
      this.event = event;
    }

    /**
     * 歩のレーンの仕事を始める。同期する歩は前の仕事の終わりを待ってから時計を始める。
     *
     * @param step 歩の計り方
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

    /** 歩の仕事の始めから最後の印までの時間（ns）。 */
    private long busy() {
      return last - begun;
    }

    /** 同期する歩なら、レーンの計算ストリームに積んだ仕事の終わりを待つ。 */
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
