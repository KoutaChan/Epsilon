package com.epsilon.agent.eval;

import com.epsilon.mahjong.rules.Placement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * 決闘の集計（{@link Duel} が使う）。牌山を番号の順に受け取り、対の量、席の分け方ごとの値、スロットごとの平均、打ち方の内訳、診断の値を
 * 足していく。続きから打つときは games.tsv の行を読み直して同じ順に足すので、止めずに打ったときと同じ値になる。同時に使うスレッドは
 * 1 つだけ。
 *
 * <p>games.tsv の 1 行は 1 半荘で、列はタブ区切りで次の順に並ぶ。
 *
 * <ol>
 *   <li>牌山の番号、割り当ての番号
 *   <li>席ごとのスロット（席の順にスロットの番号を並べた数字の列。{@code 0111} なら席 0 が候補）
 *   <li>席ごとの最終の点（席の数だけ）
 *   <li>候補の側の平均の着順（1 始まり）、候補の側の平均の効用
 *   <li>スロットごとの打ち方の数（{@link PlayStats#appendTo}）
 * </ol>
 *
 * <p>looks.tsv の 1 行は 1 つの判定の時点で、列は次の順に並ぶ。判定の時点の番号、牌山の数、半荘の数、順位の対の量とその標準誤差、効用の対の量とその
 * 標準誤差、臨界値、判定に使う量の区間の下端と上端、判定（続けるなら {@value #CONTINUE}）、止めた理由（続けるなら {@value #GOING}）、
 * 席の分け方ごとの順位の対の量（分け方の順）。
 */
final class DuelTally {

  /** looks.tsv で、続けると決めた時点の判定の列に書く値。 */
  static final String CONTINUE = "CONTINUE";

  /** looks.tsv で、続けると決めた時点の止めた理由の列に書く値。 */
  static final String GOING = "-";

  /** games.tsv で、席 0 の最終の点がある列（0 始まり）。 */
  static final int SCORE_COLUMN = 3;

  private final DuelPlan plan;
  private final ArenaPlan arena;
  private final int seats, perWall, splits;

  /** [割り当て][席] → スロット、[割り当て][スロット] → そのスロットの席の数、割り当てごとに席の順にスロットの番号を並べた文字列。 */
  private final int[][] slotOfSeat, seatsOfSlot;

  private final String[] seatSlots;
  private final double[] utilityOfPlace;
  private final PairScore pair;
  private final WallMean rank = new WallMean(), utility = new WallMean();

  /** 牌山ごとの値を集める。席の分け方ごとの順位と効用の対の量、スロットごとの着順と効用。 */
  private final WallMean[] splitRank, splitUtility, slotRank, slotUtility;

  /** [スロット][着順 − 1] の割合の和（牌山ごとの割合を足していく）。 */
  private final double[][] shares;

  /** 1 つの牌山・1 つの半荘の中で使う、スロットごとの作業領域。 */
  private final double[] wallRank, wallUtility, gameRank, gameUtility;

  /** 牌山の中の半荘ごとの局の数と、games.tsv から読んだ 1 半荘の打ち方の数。 */
  private final long[] rounds, counts;

  private final PlayStats stats;
  private int walls, identical;
  private double spread;

  DuelTally(DuelPlan plan) {
    this.plan = plan;
    arena = plan.arena();
    seats = arena.variant().seats();
    perWall = arena.gamesPerWall();
    splits = arena.splits().size();
    int slots = arena.slots().size();
    slotOfSeat = arena.slotTable();
    seatsOfSlot = new int[slotOfSeat.length][slots];
    for (int placement = 0; placement < slotOfSeat.length; placement++) {
      for (int seat = 0; seat < seats; seat++) {
        seatsOfSlot[placement][slotOfSeat[placement][seat]]++;
      }
    }
    seatSlots = seatSlots(slotOfSeat);
    utilityOfPlace = new double[seats];
    for (int place = 0; place < seats; place++) {
      utilityOfPlace[place] = arena.utility().value(arena.variant(), place);
    }
    pair = new PairScore(arena);
    splitRank = means(splits);
    splitUtility = means(splits);
    slotRank = means(slots);
    slotUtility = means(slots);
    shares = new double[slots][seats];
    wallRank = new double[slots];
    wallUtility = new double[slots];
    gameRank = new double[slots];
    gameUtility = new double[slots];
    rounds = new long[perWall];
    stats = new PlayStats(slots);
    counts = new long[stats.columns()];
  }

  private static WallMean[] means(int count) {
    WallMean[] means = new WallMean[count];
    for (int index = 0; index < count; index++) {
      means[index] = new WallMean();
    }
    return means;
  }

  /**
   * @param slotOfSeat [割り当て][席] → スロット
   * @return 割り当てごとに、席の順にスロットの番号を並べた文字列（games.tsv の席ごとのスロットの列）
   */
  static String[] seatSlots(int[][] slotOfSeat) {
    String[] texts = new String[slotOfSeat.length];
    for (int placement = 0; placement < slotOfSeat.length; placement++) {
      StringBuilder text = new StringBuilder(slotOfSeat[placement].length);
      for (int slot : slotOfSeat[placement]) {
        text.append(slot);
      }
      texts[placement] = text.toString();
    }
    return texts;
  }

  /** @return games.tsv の列の数 */
  static int columns(int seats, int slots) {
    return firstCount(seats) + new PlayStats(slots).columns();
  }

  /** @return games.tsv で、打ち方の数が始まる列（席ごとの点と、候補の側の平均の着順・効用の後ろ） */
  private static int firstCount(int seats) {
    return SCORE_COLUMN + seats + 2;
  }

  /**
   * games.tsv の 1 行を列に分け、計画のその半荘の行かを照合する。
   *
   * @param number 行の番号（0 から）
   * @param seatSlots その割り当ての、席ごとのスロットの文字列
   * @return タブで分けた列
   * @throws IllegalArgumentException 列の数・牌山の番号・割り当ての番号・席ごとのスロットのどれかが合わないとき
   */
  static String[] fields(String line, int number, long wall, int game, int placement, String seatSlots, int columns) {
    String[] fields = line.split("\t", -1);
    if (fields.length != columns
        || Long.parseLong(fields[0]) != wall
        || Integer.parseInt(fields[1]) != placement
        || !fields[2].equals(seatSlots)) {
      throw new IllegalArgumentException(
          "games.tsv line " + (number + 1) + " is not game " + game + " of wall " + wall + " with " + columns
              + " columns: " + line);
    }
    return fields;
  }

  /** @return 足した牌山の数 */
  int walls() {
    return walls;
  }

  /** @return 足した半荘の数 */
  long games() {
    return (long) walls * perWall;
  }

  /**
   * 次の牌山を足す。
   *
   * @param wall 牌山の番号（前に足した牌山の次の番号）
   * @param scores 最終の点（[牌山の中の半荘][席]）
   * @param games 半荘ごとの打ち方の内訳
   * @throws IllegalArgumentException 牌山が前に足した牌山の次でないとき
   */
  void add(long wall, int[][] scores, PlayStats[] games) {
    if (wall != arena.wallBase() + walls) {
      throw new IllegalArgumentException("Wall " + wall + " does not follow the " + walls + " walls from " + arena.wallBase());
    }
    pair.measure(wall, scores);
    rank.add(pair.rank());
    utility.add(pair.utility());
    for (int split = 0; split < splits; split++) {
      splitRank[split].add(pair.rank(split));
      splitUtility[split].add(pair.utility(split));
    }
    Arrays.fill(wallRank, 0);
    Arrays.fill(wallUtility, 0);
    boolean same = true;
    for (int game = 0; game < perWall; game++) {
      int placement = arena.placement(wall, game);
      int[] slots = slotOfSeat[placement];
      Arrays.fill(gameRank, 0);
      Arrays.fill(gameUtility, 0);
      for (int seat = 0; seat < seats; seat++) {
        int place = Placement.rank(scores[game], seat);
        int slot = slots[seat];
        gameRank[slot] += place + 1;
        gameUtility[slot] += utilityOfPlace[place];
        shares[slot][place] += 1.0 / seatsOfSlot[placement][slot] / perWall;
      }
      for (int slot = 0; slot < wallRank.length; slot++) {
        wallRank[slot] += gameRank[slot] / seatsOfSlot[placement][slot] / perWall;
        wallUtility[slot] += gameUtility[slot] / seatsOfSlot[placement][slot] / perWall;
      }
      stats.add(games[game]);
      rounds[game] = (games[game].seatHands(0) + games[game].seatHands(1)) / seats;
      same &= Arrays.equals(scores[game], scores[0]);
    }
    for (int slot = 0; slot < wallRank.length; slot++) {
      slotRank[slot].add(wallRank[slot]);
      slotUtility[slot].add(wallUtility[slot]);
    }
    if (perWall > 1) {
      if (same) {
        identical++;
      }
      spread += spread(rounds);
    }
    walls++;
  }

  /** 局の数の標準偏差（母分散の平方根）。和を整数で取るので、足す順に依らない。 */
  private static double spread(long[] rounds) {
    long sum = 0;
    long squares = 0;
    for (long count : rounds) {
      sum += count;
      squares += count * count;
    }
    long n = rounds.length;
    return Math.sqrt((double) (n * squares - sum * sum)) / n;
  }

  /**
   * 牌山の半荘ごとの行（games.tsv）を作って足す。
   *
   * @param wall 牌山の番号
   * @param scores 最終の点（[牌山の中の半荘][席]）
   * @param games 半荘ごとの打ち方の内訳
   * @param out 行を足す先
   */
  void lines(long wall, int[][] scores, PlayStats[] games, List<String> out) {
    StringBuilder line = new StringBuilder(256);
    for (int game = 0; game < perWall; game++) {
      int placement = arena.placement(wall, game);
      int[] slots = slotOfSeat[placement];
      line.setLength(0);
      line.append(wall).append('\t').append(placement).append('\t').append(seatSlots[placement]);
      double candidateRank = 0;
      double candidateUtility = 0;
      for (int seat = 0; seat < seats; seat++) {
        line.append('\t').append(scores[game][seat]);
        if (slots[seat] == 0) {
          int place = Placement.rank(scores[game], seat);
          candidateRank += place + 1;
          candidateUtility += utilityOfPlace[place];
        }
      }
      int candidates = seatsOfSlot[placement][0];
      line.append('\t').append(candidateRank / candidates).append('\t').append(candidateUtility / candidates);
      games[game].appendTo(line);
      out.add(line.toString());
    }
  }

  /**
   * games.tsv から 1 つの牌山の行を読む。
   *
   * @param lines games.tsv の行
   * @param index 牌山の添字（0 から。その牌山の行は、牌山 1 つの半荘の数 × 添字の行から始まる）
   * @param scores 点を書く先（[半荘][席]）
   * @param games 打ち方の内訳を書く先（読む前に 0 に戻す）
   * @return 牌山の番号
   * @throws IllegalArgumentException 行が計画の牌山・割り当て・席ごとのスロットと合わないとき、または列の形が正しくないとき
   */
  long parse(List<String> lines, int index, int[][] scores, PlayStats[] games) {
    long wall = arena.wallBase() + index;
    int first = firstCount(seats);
    int columns = first + counts.length;
    for (int game = 0; game < perWall; game++) {
      int number = index * perWall + game;
      int placement = arena.placement(wall, game);
      String[] fields = fields(lines.get(number), number, wall, game, placement, seatSlots[placement], columns);
      for (int seat = 0; seat < seats; seat++) {
        scores[game][seat] = Integer.parseInt(fields[SCORE_COLUMN + seat]);
      }
      for (int column = 0; column < counts.length; column++) {
        counts[column] = Long.parseLong(fields[first + column]);
      }
      games[game].clear();
      games[game].addColumns(counts);
    }
    return wall;
  }

  /**
   * 判定の時点で止めるかを決める。
   *
   * @param look 判定の時点（足した牌山の数が、その時点の牌山の数と同じであること）
   * @return 止めるならその判定。続けるなら空
   */
  Optional<Stopping.Decision> decide(Stopping stopping, int look) {
    if (walls != stopping.wallsAt(look)) {
      throw new IllegalStateException("Look " + look + " needs " + stopping.wallsAt(look) + " walls, have " + walls);
    }
    WallMean judged = judged();
    double better = plan.score() == Score.RANK ? -judged.mean() : judged.mean();
    return stopping.decide(look, better, judged.error());
  }

  /** @return 判定の時点の行（looks.tsv の 1 行） */
  String look(Stopping stopping, int look, Optional<Stopping.Decision> decision) {
    double critical = stopping.critical(look);
    WallMean judged = judged();
    StringBuilder line = new StringBuilder(160);
    line.append(look)
        .append('\t')
        .append(walls)
        .append('\t')
        .append(games())
        .append('\t')
        .append(rank.mean())
        .append('\t')
        .append(rank.error())
        .append('\t')
        .append(utility.mean())
        .append('\t')
        .append(utility.error())
        .append('\t')
        .append(critical)
        .append('\t')
        .append(judged.mean() - critical * judged.error())
        .append('\t')
        .append(judged.mean() + critical * judged.error())
        .append('\t')
        .append(decision.map(decided -> decided.verdict().name()).orElse(CONTINUE))
        .append('\t')
        .append(decision.map(decided -> decided.stopped().name()).orElse(GOING));
    for (WallMean split : splitRank) {
      line.append('\t').append(split.mean());
    }
    return line.toString();
  }

  /** @return 判定に使う量の、牌山ごとの値の平均（順位は対の量のままで、負が良い） */
  private WallMean judged() {
    return plan.score() == Score.RANK ? rank : utility;
  }

  /**
   * @param decision 止めると決めた判定
   * @param critical その判定の時点の臨界値
   * @param seconds この試行で卓を打つのにかかった秒数
   * @return 足した牌山の結果
   */
  DuelResult result(Stopping.Decision decision, double critical, double seconds) {
    List<DuelResult.SplitValue> bySplit = new ArrayList<>(splits);
    for (int split = 0; split < splits; split++) {
      bySplit.add(
          new DuelResult.SplitValue(
              arena.splits().get(split).toString(),
              Estimate.of(splitRank[split].mean(), splitRank[split].error(), critical),
              Estimate.of(splitUtility[split].mean(), splitUtility[split].error(), critical)));
    }
    List<DuelResult.SlotValue> bySlot = new ArrayList<>(slotRank.length);
    for (int slot = 0; slot < slotRank.length; slot++) {
      List<Double> placeShares = new ArrayList<>(seats);
      for (int place = 0; place < seats; place++) {
        placeShares.add(shares[slot][place] / walls);
      }
      bySlot.add(
          new DuelResult.SlotValue(
              arena.slots().get(slot).label(),
              slotRank[slot].mean(),
              slotRank[slot].error(),
              slotUtility[slot].mean(),
              slotUtility[slot].error(),
              placeShares));
    }
    return new DuelResult(
        walls,
        games(),
        decision.verdict(),
        decision.stopped(),
        plan.score(),
        critical,
        Estimate.of(rank.mean(), rank.error(), critical),
        Estimate.of(utility.mean(), utility.error(), critical),
        bySplit,
        bySlot,
        stats.rates(0).minus(stats.rates(1)),
        perWall > 1 ? identical / (double) walls : 0,
        perWall > 1 ? spread / walls : 0,
        seconds);
  }
}
