package com.epsilon.prism.collect;

import com.epsilon.agent.play.TableLoop;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * 収集の件数。判断は主経路・写し・分岐に分けて数え、主な指標の「NN 推論判断/s」は {@code modelDecisions} から求める。判断のクラスごとの
 * 配列は {@link DecisionClass} の番号で引く。手の形とクラスの配列は {@link #byHand}、根のクラスと行のクラスの配列は {@link #byRoot} の
 * 番号で引く。どのスレッドからでも加算できる。全ての判断、強制の判断、主経路の局と半荘は、卓のコアが数える（{@link TableLoop.Result}）。
 */
public final class CollectionMetrics {
  private static final int CLASSES = DecisionClass.COUNT;

  final LongAdder copyDecisions = new LongAdder();
  final LongAdder forkDecisions = new LongAdder();
  final LongAdder[] mainDecisions = adders(Openness.COUNT * CLASSES);
  final LongAdder[] labelRows = adders(Openness.COUNT * CLASSES);
  final LongAdder[] copyLabelRows = adders(CLASSES * CLASSES);
  final LongAdder[] roots = adders(CLASSES);
  final LongAdder[] copies = adders(CLASSES);
  final LongAdder[] sameDeviations = adders(CLASSES);
  final LongAdder[] deviatorDecisions = adders(CLASSES);
  final LongAdder floorRoots = new LongAdder();
  final LongAdder forks = new LongAdder();
  final LongAdder measured = new LongAdder();
  final LongAdder explored = new LongAdder();
  final LongAdder playedOutside = new LongAdder();
  final LongAdder rowsWritten = new LongAdder();
  final LongAdder mainNonGreedy = new LongAdder();

  static LongAdder[] adders(int count) {
    LongAdder[] result = new LongAdder[count];
    for (int i = 0; i < count; i++) result[i] = new LongAdder();
    return result;
  }

  /** 判断する席の手の形と判断のクラスから、配列の番号を返す。 */
  static int byHand(Openness hand, DecisionClass c) {
    return hand.ordinal() * CLASSES + c.ordinal();
  }

  /** 写しの根のクラスと行のクラスから、配列の番号を返す。 */
  static int byRoot(DecisionClass root, DecisionClass c) {
    return root.ordinal() * CLASSES + c.ordinal();
  }

  /**
   * ある時点の収集の件数。判断のクラスごとの配列は {@link DecisionClass} の番号の順に並ぶ。
   *
   * @param modelDecisions NN 推論判断の数（合法手が 2 つ以上の判断。主経路・写し・分岐の全て）
   * @param forcedDecisions 合法手が 1 つだけの強制の判断の数
   * @param hands 主経路の局の数
   * @param games 主経路の半荘の数
   * @param copyDecisions 写しの中の NN 推論判断の数（{@code modelDecisions} に含まれる）
   * @param forkDecisions 分岐の中の NN 推論判断の数（{@code modelDecisions} に含まれる）
   * @param mainDecisions クラスごとの、主経路の NN 推論判断の数（手の形は合算）
   * @param labelRows クラスごとの、主経路のラベル行の数（MAIN。手の形は合算）
   * @param copyLabelRows 行のクラスごとの、写しの中のラベル行の数（ROLLIN。根のクラスは合算）
   * @param roots クラスごとの、根の記録の数（ROOT。d = played の根も含む）
   * @param floorRoots ロールインの乱数が当たったのに、根を作らなかった判断の数。b が貪欲でない手（一様分布を混ぜた分で選んだ手）を
   *     打ったときに数える
   * @param copies 作った写しの数（d ≠ played の根の数）
   * @param sameDeviations d = played になり、写しを作らなかった根の数
   * @param forks 作った分岐の数（候補の値を測るための分岐）
   * @param measured ラベル行で値を測った手の数の和（|C ∪ {played}| の和）。行の数で割ると平均の大きさになる
   * @param explored 候補集合に一様に選んだ 1 手を足した判断の数。ラベル行か根を記録した判断ごとに 1 と数え、同じ判断のラベル行と根は
   *     合わせて 1 と数える
   * @param playedOutside 打った手が候補集合の外だったラベル行の数（一様分布を混ぜた分で選んだ手）
   * @param rowsWritten 受け取り先へ渡した行の数（MAIN + ROLLIN + ROOT）
   * @param mainNonGreedy 主経路で貪欲な手（σ の argmax）でない手を打った NN 推論判断の数（一様分布を混ぜた分と、探索の分）
   * @param breakdown 手の形と根のクラスごとの内訳
   */
  public record Snapshot(
      long modelDecisions,
      long forcedDecisions,
      long hands,
      long games,
      long copyDecisions,
      long forkDecisions,
      long[] mainDecisions,
      long[] labelRows,
      long[] copyLabelRows,
      long[] roots,
      long floorRoots,
      long copies,
      long sameDeviations,
      long forks,
      long measured,
      long explored,
      long playedOutside,
      long rowsWritten,
      long mainNonGreedy,
      Breakdown breakdown) {
    /** 全てのクラスのラベル行の数（MAIN + ROLLIN）。 */
    public long labels() {
      long total = 0;
      for (int c = 0; c < labelRows.length; c++) total += labelRows[c] + copyLabelRows[c];
      return total;
    }

    /** 全てのクラスの根の記録の数。 */
    public long rootCount() {
      long total = 0;
      for (long count : roots) total += count;
      return total;
    }

    /** クラスごとの配列（{@link #labelRows} など）を、クラスの名前から数を引く表にする。順は {@link DecisionClass} の順で、報告の jsonl に使う。 */
    public static Map<String, Long> byClass(long[] counts) {
      var result = new LinkedHashMap<String, Long>();
      for (int c = 0; c < counts.length; c++) result.put(DecisionClass.of(c).name(), counts[c]);
      return result;
    }
  }

  /**
   * 収集の行の内訳。報告の jsonl に使う。表は手の形とクラスの順に並び、0 の項目と空の表は載せない。
   *
   * @param decisions 判断する席の手の形 → クラス → 主経路の NN 推論判断の数
   * @param labels 判断する席の手の形 → クラス → 主経路のラベル行の数
   * @param copies 根のクラス → 写しを作った根の数（d ≠ played）
   * @param sameDeviations 根のクラス → d = played になり、写しを作らなかった根の数
   * @param deviatorDecisions 根のクラス → 写しの中で逸脱した席の NN 推論判断のうち、結果の違う合法手が 2 つ以上あるものの数。
   *     写しのラベル行の候補にあたり、行の数が上限に達した後も数える
   * @param copyLabels 根のクラス → 写しの中のラベル行のクラス → 行の数
   */
  public record Breakdown(
      Map<Openness, Map<DecisionClass, Long>> decisions,
      Map<Openness, Map<DecisionClass, Long>> labels,
      Map<DecisionClass, Long> copies,
      Map<DecisionClass, Long> sameDeviations,
      Map<DecisionClass, Long> deviatorDecisions,
      Map<DecisionClass, Map<DecisionClass, Long>> copyLabels) {

    /** 副露している席の、主経路のラベル行の数。 */
    public long openLabels() {
      Map<DecisionClass, Long> open = labels.get(Openness.OPEN);
      long total = 0;
      if (open != null) for (long count : open.values()) total += count;
      return total;
    }
  }

  /**
   * 今の件数を読んで返す。
   *
   * @param table 同じ収集での卓の結果（全ての判断、強制の判断、主経路の局と半荘の数を読む）
   * @return 件数
   */
  public Snapshot snapshot(TableLoop.Result table) {
    long[] decisionsByHand = sums(mainDecisions);
    long[] labelsByHand = sums(labelRows);
    long[] copyLabelsByRoot = sums(copyLabelRows);
    long[] copiesByRoot = sums(copies);
    long[] sameByRoot = sums(sameDeviations);
    return new Snapshot(
        table.queriedDecisions(),
        table.forcedDecisions(),
        table.rounds(),
        table.games(),
        copyDecisions.sum(),
        forkDecisions.sum(),
        sumBlocks(decisionsByHand, Openness.COUNT),
        sumBlocks(labelsByHand, Openness.COUNT),
        sumBlocks(copyLabelsByRoot, CLASSES),
        sums(roots),
        floorRoots.sum(),
        total(copiesByRoot),
        total(sameByRoot),
        forks.sum(),
        measured.sum(),
        explored.sum(),
        playedOutside.sum(),
        rowsWritten.sum(),
        mainNonGreedy.sum(),
        new Breakdown(
            handTable(decisionsByHand),
            handTable(labelsByHand),
            classTable(copiesByRoot, 0),
            classTable(sameByRoot, 0),
            classTable(sums(deviatorDecisions), 0),
            rootTable(copyLabelsByRoot)));
  }

  static long[] sums(LongAdder[] adders) {
    long[] result = new long[adders.length];
    for (int i = 0; i < adders.length; i++) result[i] = adders[i].sum();
    return result;
  }

  /** クラスの数ずつの区切りが {@code blocks} 個並んだ配列を、クラスごとに足し合わせる。 */
  private static long[] sumBlocks(long[] counts, int blocks) {
    long[] result = new long[CLASSES];
    for (int block = 0; block < blocks; block++)
      for (int c = 0; c < CLASSES; c++) result[c] += counts[block * CLASSES + c];
    return result;
  }

  private static long total(long[] counts) {
    long total = 0;
    for (long count : counts) total += count;
    return total;
  }

  /** {@code offset} から始まるクラスの数ぶんの項目を、0 でないクラスだけ載せた表にする。表はクラスの順に並ぶ。 */
  private static Map<DecisionClass, Long> classTable(long[] counts, int offset) {
    Map<DecisionClass, Long> table = new EnumMap<>(DecisionClass.class);
    for (DecisionClass c : DecisionClass.values()) {
      long count = counts[offset + c.ordinal()];
      if (count > 0) table.put(c, count);
    }
    return Collections.unmodifiableMap(table);
  }

  /** {@link #byHand} の番号で並んだ配列を、手の形 → クラス → 数の表にする。 */
  private static Map<Openness, Map<DecisionClass, Long>> handTable(long[] counts) {
    Map<Openness, Map<DecisionClass, Long>> table = new EnumMap<>(Openness.class);
    for (Openness hand : Openness.values()) {
      Map<DecisionClass, Long> row = classTable(counts, hand.ordinal() * CLASSES);
      if (!row.isEmpty()) table.put(hand, row);
    }
    return Collections.unmodifiableMap(table);
  }

  /** {@link #byRoot} の番号で並んだ配列を、根のクラス → 行のクラス → 数の表にする。 */
  private static Map<DecisionClass, Map<DecisionClass, Long>> rootTable(long[] counts) {
    Map<DecisionClass, Map<DecisionClass, Long>> table = new EnumMap<>(DecisionClass.class);
    for (DecisionClass root : DecisionClass.values()) {
      Map<DecisionClass, Long> row = classTable(counts, root.ordinal() * CLASSES);
      if (!row.isEmpty()) table.put(root, row);
    }
    return Collections.unmodifiableMap(table);
  }
}
