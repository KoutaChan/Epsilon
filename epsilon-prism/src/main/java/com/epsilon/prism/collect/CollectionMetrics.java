package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * 収集の件数。判断は役割別に数え、主指標の「NN 推論判断/s」は modelDecisions から求める。類ごとの配列は {@link DecisionClass} の番号で引く。
 * どのスレッドからでも加算できる。
 */
public final class CollectionMetrics {
  final LongAdder modelDecisions = new LongAdder();
  final LongAdder forcedDecisions = new LongAdder();
  final LongAdder hands = new LongAdder();
  final LongAdder games = new LongAdder();
  final LongAdder copyDecisions = new LongAdder();
  final LongAdder forkDecisions = new LongAdder();
  final LongAdder[] mainDecisions = adders(DecisionClass.COUNT);
  final LongAdder[] labelRows = adders(DecisionClass.COUNT);
  final LongAdder[] copyLabelRows = adders(DecisionClass.COUNT);
  final LongAdder[] roots = adders(DecisionClass.COUNT);
  final LongAdder floorRoots = new LongAdder();
  final LongAdder copies = new LongAdder();
  final LongAdder sameDeviations = new LongAdder();
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

  /**
   * 収集の件数の写し。類ごとの配列は {@link DecisionClass} の番号の順。
   *
   * @param modelDecisions NN 推論判断（合法手 2 つ以上。主経路・写し・分岐の全て）
   * @param forcedDecisions 合法手 1 つの強制手
   * @param hands 主経路の局の数
   * @param games 主経路の半荘の数
   * @param copyDecisions 写しの中の NN 推論判断（modelDecisions に含む）
   * @param forkDecisions 分岐の中の NN 推論判断（modelDecisions に含む）
   * @param mainDecisions 類ごとの主経路の NN 推論判断
   * @param labelRows 類ごとの主経路のラベル行（MAIN）
   * @param copyLabelRows 類ごとの写しの中のラベル行（ROLLIN）
   * @param roots 類ごとの根の記録（ROOT。d = played を含む）
   * @param floorRoots ロールインの硬貨が当たったが、b が床 ε_b で貪欲でない手を打ったので根を作らなかった判断
   * @param copies 作った写し（d ≠ played の根の数）
   * @param sameDeviations d = played で写しを作らなかった根
   * @param forks 作った分岐（候補の測定）
   * @param measured ラベル行の測った手の数の和（|C ∪ {played}| の和。行の数で割ると平均の大きさ）
   * @param explored 候補集合に一様な 1 手を足した判断（ラベル行か根を記録した判断ごとに 1。同じ判断のラベル行と根は 1 と数える）
   * @param playedOutside 打った手が候補集合の外だった（床 ε_b の手）ラベル行
   * @param rowsWritten 受け口へ渡した行（MAIN + ROLLIN + ROOT）
   * @param mainNonGreedy 主経路で貪欲（σ の argmax）でない手を打った NN 推論判断（床 ε_b と探索の分）
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
      long mainNonGreedy) {
    /** 全ての類のラベル行（MAIN + ROLLIN）。 */
    public long labels() {
      long total = 0;
      for (int c = 0; c < labelRows.length; c++) total += labelRows[c] + copyLabelRows[c];
      return total;
    }

    /** 全ての類の根の記録。 */
    public long rootCount() {
      long total = 0;
      for (long count : roots) total += count;
      return total;
    }

    /** 類ごとの配列（{@link #labelRows} など）を、類の名前 → 数の表にする（{@link DecisionClass} の順。報告の jsonl 用）。 */
    public static Map<String, Long> byClass(long[] counts) {
      var result = new LinkedHashMap<String, Long>();
      for (int c = 0; c < counts.length; c++) result.put(DecisionClass.of(c).name(), counts[c]);
      return result;
    }
  }

  public Snapshot snapshot() {
    return new Snapshot(
        modelDecisions.sum(),
        forcedDecisions.sum(),
        hands.sum(),
        games.sum(),
        copyDecisions.sum(),
        forkDecisions.sum(),
        sums(mainDecisions),
        sums(labelRows),
        sums(copyLabelRows),
        sums(roots),
        floorRoots.sum(),
        copies.sum(),
        sameDeviations.sum(),
        forks.sum(),
        measured.sum(),
        explored.sum(),
        playedOutside.sum(),
        rowsWritten.sum(),
        mainNonGreedy.sum());
  }

  static long[] sums(LongAdder[] adders) {
    long[] result = new long[adders.length];
    for (int i = 0; i < adders.length; i++) result[i] = adders[i].sum();
    return result;
  }
}
