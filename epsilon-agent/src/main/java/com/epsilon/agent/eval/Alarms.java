package com.epsilon.agent.eval;

import com.epsilon.agent.eval.Alarm.Handling;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.OptionalInt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * リグレッションのアラート。王者との決闘、「悪い」の連続、ラダー、低下のアラートをここで決め、{@link #write} で run のアラートの行
 * （reports/alarms.jsonl）に 1 度ずつ書く。値は全て、大きいほど候補が良い向きで受け取る（順位なら符号を反転する）。王者は、王者にした
 * バージョンの番号で表す。
 *
 * <p>低下のアラートは、同じ卓で同じ王者の間は 1 回だけ出す。どの王者で出したかはアラートの行から読むので、落ちた後に当て直しても、別の
 * JVM で続けても同じに決まる。
 */
public final class Alarms {

  private static final Logger LOG = LoggerFactory.getLogger(Alarms.class);

  private final Watch watch;
  private final Path file;

  /**
   * @param watch 監視の設定
   * @param run アラートの行を書く run
   */
  public Alarms(Watch watch, RunDirectory run) {
    this.watch = Objects.requireNonNull(watch, "watch");
    this.file = run.report(Alarm.FILE);
  }

  /**
   * 王者との決闘の判定を 1 つ当てた後の、「悪い」が続いた回数。
   *
   * @param before 前の決闘までに悪いが続いた回数
   * @param verdict この決闘の判定
   * @return 悪いなら 1 増やした数、ほかの判定なら 0
   */
  public static int streak(int before, Verdict verdict) {
    if (before < 0) {
      throw new IllegalArgumentException("Worse streak must be non-negative: " + before);
    }
    return verdict == Verdict.WORSE ? before + 1 : 0;
  }

  /**
   * 悪いが続いた回数を数えない run（evaluate）の、王者の決闘のアラート。
   *
   * @param job 決闘のジョブの名前
   * @param version 候補のバージョン
   * @param verdict 決闘の判定
   * @param better 決闘の平均と区間
   * @return 判定が「悪い」なら、王者より悪いというアラート
   */
  public List<Alarm> championDuel(String job, int version, Verdict verdict, Estimate better) {
    return verdict == Verdict.WORSE
        ? List.of(Alarm.of(job, version, Signal.CHAMPION_WORSE, better, Handling.REPORT, null))
        : List.of();
  }

  /**
   * 王者の決闘のアラート。悪いなら王者に悪いを出し、悪いが設定の回数以上続いていれば、悪いが続くも出す。悪いが続くの扱いは設定の形で
   * 決まる。王者へ戻す形は、戻しを数える幅の中でもう上限まで戻していれば、一時停止に替える。
   *
   * @param job 決闘のジョブの名前
   * @param version 候補のバージョン
   * @param verdict 決闘の判定
   * @param better 決闘の平均と区間
   * @param streak この決闘を当てた後の、悪いが続いた回数（{@link #streak}）
   * @param rollbacks これまでに王者へ戻した決闘のバージョン
   * @return 王者に悪い、悪いが続くの順
   * @throws IllegalArgumentException 続いた回数が判定に合わないとき（悪いなら 1 以上、ほかの判定なら 0）
   */
  public List<Alarm> championDuel(
      String job, int version, Verdict verdict, Estimate better, int streak, List<Integer> rollbacks) {
    if (verdict == Verdict.WORSE ? streak < 1 : streak != 0) {
      throw new IllegalArgumentException("A " + verdict + " duel cannot leave a worse streak of " + streak);
    }
    List<Alarm> alarms = new ArrayList<>(championDuel(job, version, verdict, better));
    OnWorseSettings onWorse = watch.onWorse();
    if (verdict == Verdict.WORSE && streak >= onWorse.streak()) {
      Handling handling =
          switch (onWorse) {
            case OnWorseSettings.Report report -> Handling.REPORT;
            case OnWorseSettings.Pause pause -> Handling.PAUSE;
            case OnWorseSettings.Rollback rollback ->
                rolledBack(rollbacks, version, rollback.span()) >= rollback.maxRollbacks()
                    ? Handling.PAUSE
                    : Handling.ROLLBACK;
          };
      alarms.add(Alarm.of(job, version, Signal.WORSE_STREAK, better, handling, null));
    }
    return alarms;
  }

  /** 戻しを数える幅の中で王者へ戻した回数。幅は決闘のバージョンで終わり、そこから幅の分だけ前のバージョンは数えない。 */
  private static long rolledBack(List<Integer> rollbacks, int version, int span) {
    return rollbacks.stream().filter(rolled -> rolled > version - span && rolled <= version).count();
  }

  /**
   * ラダーのアラート。基準の卓の結果で判定するラダーも、ここで出す。
   *
   * @param job 卓のジョブの名前
   * @param version 候補のバージョン
   * @param better ラダーの卓の平均と、固定の有意水準の区間
   * @return 区間の上端が 0 より小さければ、ラダーで悪いというアラート
   */
  public List<Alarm> ladder(String job, int version, Estimate better) {
    return better.high() < 0
        ? List.of(Alarm.of(job, version, Signal.LADDER_WORSE, better, Handling.REPORT, null))
        : List.of();
  }

  /**
   * 基準の卓の値を、王者の同じ卓の値と比べる。対の差の平均が −幅 以下で、悪い向きの z がしきい値を超えれば、基準の卓の低下のアラートを
   * 出す。
   *
   * @param job 基準の卓のジョブの名前（卓の名前はジョブの名前から読む）
   * @param version 今のバージョン
   * @param champion 今の王者のバージョン
   * @param current 今のバージョンの、牌山ごとの値
   * @param championValues 王者の同じ卓の、牌山ごとの値（同じ牌山の集合）
   * @return 基準の卓の低下のアラート。同じ卓で同じ王者の低下をもう書いていれば空
   */
  public List<Alarm> baseline(String job, int version, int champion, double[] current, double[] championValues)
      throws IOException {
    return drop(job, version, Signal.BASELINE_DROP, champion, current, championValues, watch.margin());
  }

  /**
   * ルールベースの打ち手の卓の値を、王者の同じ卓の値と比べる。対の差の悪い向きの z がしきい値を超えれば、低下を出す（幅は使わない）。
   *
   * @param job 卓のジョブの名前（卓の名前はジョブの名前から読む）
   * @param version 今のバージョン
   * @param champion 今の王者のバージョン
   * @param current 今のバージョンの、牌山ごとの値
   * @param championValues 王者の同じ卓の、牌山ごとの値（同じ牌山の集合）
   * @return ルールベースの打ち手の卓の低下。同じ卓で同じ王者の低下をもう書いていれば空
   */
  public List<Alarm> anchor(String job, int version, int champion, double[] current, double[] championValues)
      throws IOException {
    return drop(job, version, Signal.ANCHOR_DROP, champion, current, championValues, 0);
  }

  private List<Alarm> drop(
      String job, int version, Signal signal, int champion, double[] current, double[] championValues, double margin)
      throws IOException {
    OptionalInt raised = lastChampion(signal, job);
    if (raised.isPresent() && raised.getAsInt() == champion) {
      return List.of();
    }
    WallMean difference = PairedWalls.difference(current, championValues);
    double mean = difference.mean();
    double error = difference.error();
    boolean worse = mean < 0 && -mean >= margin && -mean / error > watch.alarmZ();
    if (!worse) {
      return List.of();
    }
    return List.of(
        Alarm.of(job, version, signal, Estimate.of(mean, error, watch.alarmZ()), Handling.REPORT, champion));
  }

  /**
   * ジョブの卓で、低下のアラートを最後に書いたときの王者。
   *
   * @return 王者のバージョン。まだ書いていなければ空
   */
  private OptionalInt lastChampion(Signal signal, String job) throws IOException {
    String table = table(job);
    OptionalInt champion = OptionalInt.empty();
    for (Alarm line : read()) {
      boolean sameTable =
          RunDirectory.parseJobName(line.job()).map(name -> name.table().equals(table)).orElse(false);
      if (line.signal() == signal && line.champion() != null && sameTable) {
        champion = OptionalInt.of(line.champion());
      }
    }
    return champion;
  }

  private static String table(String job) {
    return RunDirectory.parseJobName(job)
        .orElseThrow(() -> new IllegalArgumentException("Not the name of a table job: " + job))
        .table();
  }

  /**
   * アラートを書く。同じ名前の行がもうあれば書かない（落ちた後に当て直しても 1 度と同じ）。
   *
   * @param alarms 書くアラート（この順に書く）
   */
  public void write(List<Alarm> alarms) throws IOException {
    for (Alarm alarm : alarms) {
      if (DurableFiles.appendOnce(file, "alarm", new Node.Text(alarm.alarm(), "alarm"), RecordFile.tree(alarm))) {
        LOG.warn(
            "Alarm {} at version {}: {} [{}, {}] ({})",
            alarm.signal(),
            alarm.version(),
            String.format(Locale.ROOT, "%.4f", alarm.estimate().mean()),
            String.format(Locale.ROOT, "%.4f", alarm.estimate().low()),
            String.format(Locale.ROOT, "%.4f", alarm.estimate().high()),
            alarm.handling());
      }
    }
  }

  /** @return 書いたアラート（書いた順）。まだ無ければ空 */
  public List<Alarm> read() throws IOException {
    List<String> lines = DurableFiles.lines(file);
    List<Alarm> alarms = new ArrayList<>(lines.size());
    for (int index = 0; index < lines.size(); index++) {
      alarms.add(RecordFile.read(Alarm.class, Json.read(lines.get(index), file + ":" + (index + 1))).value());
    }
    return alarms;
  }
}
