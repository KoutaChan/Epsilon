package com.epsilon.classic.learn;

import com.epsilon.agent.settings.Key;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 方策の学習率を、macro ごとに測った KL(π_前 ‖ π_今) に合わせて動かす。macro は、収集と学習を 1 回ずつ行う反復。macro の KL は、学習の
 * ステップごとに更新の前に測った値の平均（{@link SelfPlayLearner}）。
 *
 * <ol>
 *   <li>ウォームアップ: 最初の {@code warmup} 回の macro は、初めの学習率のまま KL を測る。測り終えたら、基準をその中央値にし、目標を
 *       基準 × {@code targetMultiplier} にする。
 *   <li>目標の推移: 受け入れた方策のステップの累計が予定のステップ数の 1 割になるまで（ウォームアップがそれより長ければウォームアップの
 *       終わりまで）、目標を保つ。その後、予定のステップ数で {@code finalScale} 倍になるまで cos で下げ、それより後は保つ。予定の
 *       ステップ数が 0 なら目標を下げない。
 *   <li>制御: 直近 {@code window} 回の KL の中央値 m が [目標/b, 目標·b]（b は {@code deadBand}）の外にあれば、学習率に √(目標/m) を
 *       掛ける。1 回に動かすのは {@code maximumFactor} 倍まで。学習率は [{@code minimum}, {@code maximum}] に収める。
 * </ol>
 *
 * 状態（{@link State}）は record なので、そのまま書き出して読み戻せる。
 */
public final class KlLearningRate {

  /** 割る数にする KL の中央値の下限。中央値が 0 でも割れるようにする。 */
  private static final double MINIMUM_KL = 1e-12;

  /** 目標を保つ期間は、予定のステップ数をこの値で割った長さ（切り上げ）。 */
  private static final long HOLD_DIVISOR = 10;

  /**
   * 制御の設定。既定は Nano の自己対局の値。
   *
   * @param initial 初めの学習率（ウォームアップの間もこの値）
   * @param minimum 学習率の下限
   * @param maximum 学習率の上限
   * @param warmup 基準を測る macro の数
   * @param window 中央値を取る直近の macro の数
   * @param targetMultiplier 基準に掛けて初めの目標にする倍率（1 以上）
   * @param deadBand 学習率を動かさない幅の倍率（1 より大きい）
   * @param maximumFactor 1 回の macro で学習率を動かす倍率の上限（1 より大きい）
   * @param plannedSteps 目標を下げ終えるときの、受け入れた方策のステップの累計（0 なら目標を下げない）
   * @param finalScale 下げ終えた目標が初めの目標の何倍か（0 より大きく 1 以下）
   */
  public record Settings(
      @Key(doc = "初めの学習率（ウォームアップの間もこの値）", value = "1.25e-6") double initial,
      @Key(doc = "学習率の下限", value = "2.5e-7") double minimum,
      @Key(doc = "学習率の上限", value = "5e-6") double maximum,
      @Key(doc = "基準を測る macro の数", value = "3") int warmup,
      @Key(doc = "中央値を取る直近の macro の数", value = "3") int window,
      @Key(doc = "基準に掛けて初めの目標にする倍率（1 以上）", value = "1.25") double targetMultiplier,
      @Key(doc = "学習率を動かさない幅の倍率（1 より大きい）", value = "1.25") double deadBand,
      @Key(doc = "1 回の macro で学習率を動かす倍率の上限（1 より大きい）", value = "1.02") double maximumFactor,
      @Key(doc = "目標を下げ終えるときの、受け入れた方策のステップの累計（0 なら目標を下げない）", value = "43200") long plannedSteps,
      @Key(doc = "下げ終えた目標が初めの目標の何倍か（0 より大きく 1 以下）", value = "0.5") double finalScale) {

    public Settings {
      if (!(minimum > 0) || !(minimum <= initial) || !(initial <= maximum) || !Double.isFinite(maximum)) {
        throw new IllegalArgumentException(
            "Learning rates must satisfy 0 < minimum <= initial <= maximum: " + minimum + ", " + initial + ", " + maximum);
      }
      if (warmup < 1 || window < 1 || plannedSteps < 0) {
        throw new IllegalArgumentException("Bad KL control windows: " + warmup + ", " + window + ", " + plannedSteps);
      }
      if (!(targetMultiplier >= 1) || !(deadBand > 1) || !(maximumFactor > 1) || !(finalScale > 0 && finalScale <= 1)
          || !Double.isFinite(targetMultiplier) || !Double.isFinite(deadBand) || !Double.isFinite(maximumFactor)) {
        throw new IllegalArgumentException("Bad KL control factors: " + this);
      }
    }
  }

  /**
   * 目標の KL のスケジュール。ステップは、受け入れた方策のステップの累計で数える。
   *
   * @param anchorStep ウォームアップを終えたステップ
   * @param anchorTarget 初めの目標
   * @param decayStep 目標を下げ始めるステップ
   * @param endStep 目標を下げ終えるステップ
   * @param endTarget 下げ終えた目標
   */
  public record Schedule(long anchorStep, double anchorTarget, long decayStep, long endStep, double endTarget) {

    public Schedule {
      if (anchorStep < 0 || decayStep < anchorStep || endStep < decayStep) {
        throw new IllegalArgumentException("KL schedule steps must be ordered: " + anchorStep + ", " + decayStep + ", " + endStep);
      }
      if (!(anchorTarget > 0) || !(endTarget > 0) || !(endTarget <= anchorTarget) || !Double.isFinite(anchorTarget)) {
        throw new IllegalArgumentException("KL schedule targets must be positive and non-increasing: " + anchorTarget + ", " + endTarget);
      }
    }

    static Schedule of(long step, double target, Settings settings) {
      long planned = settings.plannedSteps();
      if (planned == 0) {
        return new Schedule(step, target, step, step, target);
      }
      if (planned <= step) {
        throw new IllegalStateException("The KL target plan of " + planned + " steps ended before the warm-up at " + step);
      }
      long hold = Math.max(step, (planned + HOLD_DIVISOR - 1) / HOLD_DIVISOR);
      return new Schedule(step, target, hold, planned, target * settings.finalScale());
    }

    /** そのステップでの目標。 */
    public double target(long step) {
      if (step >= endStep) {
        return endTarget;
      }
      if (step <= decayStep) {
        return anchorTarget;
      }
      double progress = (double) (step - decayStep) / (endStep - decayStep);
      return endTarget + (anchorTarget - endTarget) * 0.5 * (1 + Math.cos(Math.PI * progress));
    }
  }

  /**
   * 制御の状態。
   *
   * @param macros 受け入れた macro の数
   * @param steps 受け入れた方策のステップの累計
   * @param rate 次の macro の学習率
   * @param warmup ウォームアップで測った KL（macro の順）
   * @param recent 直近の macro の KL（古い順、{@code window} 個まで）
   * @param baseline ウォームアップで測った KL の中央値（ウォームアップの間は 0）
   * @param schedule 目標のスケジュール（ウォームアップの間は null）
   */
  public record State(
      int macros, long steps, double rate, List<Double> warmup, List<Double> recent, double baseline, Schedule schedule) {

    public State {
      warmup = List.copyOf(warmup);
      recent = List.copyOf(recent);
      if (macros < 0 || steps < 0 || !(rate > 0) || !Double.isFinite(rate)) {
        throw new IllegalArgumentException("Bad KL control state: " + macros + ", " + steps + ", " + rate);
      }
      for (double kl : warmup) {
        requireKl(kl);
      }
      for (double kl : recent) {
        requireKl(kl);
      }
      if ((schedule == null) != (baseline == 0)) {
        throw new IllegalArgumentException("A KL control state has a baseline exactly when it has a schedule");
      }
    }
  }

  /** 1 回の macro で学習率をどう動かしたか。 */
  public enum Action {
    /** ウォームアップの間。学習率は初めの値のまま。 */
    WARMUP,
    /** 中央値が幅の中にあり、学習率を動かさない。 */
    HOLD,
    INCREASE,
    DECREASE,
    /** 下げて下限に当たった。 */
    AT_MINIMUM,
    /** 上げて上限に当たった。 */
    AT_MAXIMUM,
    /** 受け入れなかった macro。状態は変えない。 */
    REJECTED
  }

  /**
   * 1 回の macro の記録。
   *
   * @param applied その macro に使った学習率
   * @param observed その macro の KL
   * @param median 直近の KL の中央値
   * @param target その時点の目標（ウォームアップの間は NaN）
   * @param next 次の macro の学習率
   */
  public record Adjustment(Action action, double applied, double observed, double median, double target, double next) {}

  private final Settings settings;
  private State state;

  /** ウォームアップから始める。 */
  public KlLearningRate(Settings settings) {
    this.settings = Objects.requireNonNull(settings, "settings");
    state = new State(0, 0, settings.initial(), List.of(), List.of(), 0, null);
  }

  /**
   * 書き出した状態から続ける。
   *
   * @throws IllegalArgumentException 状態がこの設定と合わないとき（学習率が範囲の外にある、ウォームアップや直近の KL の数が macro の数と
   *     合わない、ウォームアップを終えたのにスケジュールが無いか、終える前なのにスケジュールがある）
   */
  public KlLearningRate(Settings settings, State state) {
    this.settings = Objects.requireNonNull(settings, "settings");
    Objects.requireNonNull(state, "state");
    if (state.rate() < settings.minimum() || state.rate() > settings.maximum()) {
      throw new IllegalArgumentException("The saved learning rate " + state.rate() + " is outside " + settings);
    }
    boolean warmed = state.macros() >= settings.warmup();
    if (state.warmup().size() != Math.min(state.macros(), settings.warmup())
        || state.recent().size() != Math.min(state.macros(), settings.window())
        || (state.schedule() != null) != warmed) {
      throw new IllegalArgumentException("The saved KL control state does not match " + settings + ": " + state);
    }
    this.state = state;
  }

  public Settings settings() {
    return settings;
  }

  /** 次の macro の学習率。 */
  public double rate() {
    return state.rate();
  }

  public State state() {
    return state;
  }

  /**
   * 学習を受け入れた macro の KL を足し、次の macro の学習率を決める。
   *
   * @param kl macro の KL（有限で 0 以上）
   * @param acceptedSteps その macro で受け入れた方策のステップ数（1 以上）
   */
  public Adjustment accept(double kl, int acceptedSteps) {
    requireKl(kl);
    if (acceptedSteps < 1) {
      throw new IllegalArgumentException("An accepted macro takes at least one policy step: " + acceptedSteps);
    }
    State before = state;
    long steps = Math.addExact(before.steps(), acceptedSteps);
    double applied = before.rate();
    List<Double> recent = appended(before.recent(), kl, settings.window());
    double median = median(recent);
    if (before.schedule() == null) {
      List<Double> warmup = appended(before.warmup(), kl, settings.warmup());
      double baseline = 0;
      Schedule schedule = null;
      if (warmup.size() == settings.warmup()) {
        baseline = Math.max(MINIMUM_KL, median(warmup));
        schedule = Schedule.of(steps, baseline * settings.targetMultiplier(), settings);
      }
      state = new State(before.macros() + 1, steps, applied, warmup, recent, baseline, schedule);
      return new Adjustment(
          Action.WARMUP, applied, kl, median, schedule == null ? Double.NaN : schedule.target(steps), applied);
    }
    double target = before.schedule().target(steps);
    double lower = target / settings.deadBand();
    double upper = target * settings.deadBand();
    Action action = Action.HOLD;
    double next = applied;
    if (median < lower || median > upper) {
      double raw = applied * Math.sqrt(target / Math.max(median, Math.min(MINIMUM_KL, lower)));
      next = Math.max(applied / settings.maximumFactor(), Math.min(applied * settings.maximumFactor(), raw));
      action = median < lower ? Action.INCREASE : Action.DECREASE;
      if (next > settings.maximum()) {
        next = settings.maximum();
        action = Action.AT_MAXIMUM;
      } else if (next < settings.minimum()) {
        next = settings.minimum();
        action = Action.AT_MINIMUM;
      }
    }
    state =
        new State(before.macros() + 1, steps, next, before.warmup(), recent, before.baseline(), before.schedule());
    return new Adjustment(action, applied, kl, median, target, next);
  }

  /**
   * 学習を受け入れなかった macro の KL を、記録にだけ残す。状態と学習率は変えない。
   *
   * @param kl macro の KL（有限で 0 以上）
   */
  public Adjustment reject(double kl) {
    requireKl(kl);
    double target = state.schedule() == null ? Double.NaN : state.schedule().target(state.steps());
    double median = state.recent().isEmpty() ? Double.NaN : median(state.recent());
    return new Adjustment(Action.REJECTED, state.rate(), kl, median, target, state.rate());
  }

  private static List<Double> appended(List<Double> values, double value, int limit) {
    List<Double> next = new ArrayList<>(values);
    next.add(value);
    while (next.size() > limit) {
      next.removeFirst();
    }
    return next;
  }

  private static double median(List<Double> values) {
    List<Double> sorted = new ArrayList<>(values);
    Collections.sort(sorted);
    int middle = sorted.size() / 2;
    return sorted.size() % 2 == 1 ? sorted.get(middle) : (sorted.get(middle - 1) + sorted.get(middle)) / 2;
  }

  private static void requireKl(double kl) {
    if (!(kl >= 0) || !Double.isFinite(kl)) {
      throw new IllegalArgumentException("A KL must be finite and non-negative: " + kl);
    }
  }
}
