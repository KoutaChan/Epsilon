package com.epsilon.prism.luck;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WallReplay;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 運の精算（{@link LedgerTracker}）が守る法則を確かめる。偶然のイベントで出る牌は、誰からも見えない牌のプールの上で一様に出る。
 * 本物の牌山では PIT で、牌譜の局では読んでいない位置を並べ替えて数えて確かめる。局面を固定して読んでいない位置だけを並べ替えると、
 * 帳簿の増分の平均は局面ごとに 0 になる。これで、イベントの前には分からないことを c が読んでしまう漏れを捕まえる。局末の裏ドラの項も
 * 席ごとに確かめる。ここでは小さく回す。同じ検定を大きく回すのは {@link LedgerLawBench}。
 */
public class LedgerLawTest {
  private static final int TILES = Tiles.KINDS;

  private static final LedgerColumn[] COLUMNS = LedgerColumn.values();

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants", timeOut = 600_000)
  public void drawsAreUniformOverThePool(Variant variant) {
    drawsAreUniform(variant, variant.seats() == 4 ? 8000 : 6000);
  }

  @Test(dataProvider = "variants", timeOut = 600_000)
  public void poolMatchesTheUnseenSet(Variant variant) {
    poolMatches(variant, 3, 2, 2000);
  }

  /** ルールと、点の列の単位が違う帳簿の組。 */
  @DataProvider
  public Object[][] units() {
    return new Object[][] {
      {Variant.TENHOU_4P, LedgerGames.RECORDING},
      {Variant.TENHOU_3P, LedgerGames.RECORDING},
      {Variant.TENHOU_4P, LedgerGames.PHI_RECORDING},
      {Variant.TENHOU_3P, LedgerGames.PHI_RECORDING},
      {Variant.TENHOU_4P, LedgerGames.PHI_DRAWS_RECORDING},
      {Variant.TENHOU_3P, LedgerGames.PHI_DRAWS_RECORDING}
    };
  }

  @Test(dataProvider = "units", timeOut = 600_000)
  public void conditionalMeanIsZeroUnderReshuffle(Variant variant, LedgerRecording recording) {
    conditionalMean(variant, recording, new int[] {2, 2, 2, 2}, 1500);
  }

  @Test(dataProvider = "variants", timeOut = 600_000)
  public void phiPointColumnsVanishOnADecidedTable(Variant variant) {
    decidedTable(variant, 60);
  }

  @Test(dataProvider = "variants", timeOut = 600_000)
  public void uraTermIsCentredUnderReshuffle(Variant variant) {
    uraCentred(variant, 2, 1000);
  }

  /** PIT のヒストグラムのビンの数。 */
  private static final int BINS = 20;

  /** 乱択化した PIT のヒストグラム。20 個のビンの度数（{@link #add} で数える）と件数を持つ。 */
  static final class Pit {
    final String name;
    final long[] bins = new long[BINS];
    long count;

    Pit(String name) {
      this.name = name;
    }

    /**
     * 種類 {@code tile} の牌が出たことを数える。{@code pool} は出る前のプール、{@code v} は独立な一様乱数で、
     * u = (Σ_{t<tile} n_t + v·n_tile) / N が入るビンに 1 を足す。
     */
    void add(int[] pool, int tile, double v) {
      int below = 0, total = 0;
      for (int t = 0; t < TILES; t++) {
        if (t < tile) below += pool[t];
        total += pool[t];
      }
      Assert.assertTrue(pool[tile] > 0, name + ": tile " + tile + " outside the pool");
      double u = (below + v * pool[tile]) / total;
      bins[Math.min(BINS - 1, (int) (u * BINS))]++;
      count++;
    }

    double p() {
      double[] expected = new double[BINS];
      Arrays.fill(expected, count / (double) BINS);
      return LedgerGames.chiSquareUpper(LedgerGames.chiSquare(bins, expected), BINS - 1);
    }

    void check(long least) {
      Assert.assertTrue(count >= least, name + " count " + count);
      if (count >= 200) Assert.assertTrue(p() > 1e-4, name + " p=" + p() + " n=" + count);
    }
  }

  /**
   * 本物の牌山（席ごとのツモの列）で局を多く打ち、全ての偶然のイベントを乱択化した PIT にかけて、20 個のビンのカイ二乗で検定する。
   * PIT に使うプールはイベントの前のもので、局の始めの枚数から前のイベントで出た牌を引いて作る。検定は次の 5 つに分けて行う: 山のツモ、
   * 嶺上と抜きの補充、席の列が尽きた後のツモ、槓ドラの表示、裏ドラ。席の列は配牌の後の山を席の順に配ったもので、嶺上と補充も自分の列から
   * 引く。列が尽きた後のツモは、席のツモの回数が列の長さを超えたもの。
   */
  static void drawsAreUniform(Variant variant, int rounds) {
    var pitRandom = new Random(20261028L + variant.ordinal());
    Pit live = new Pit(variant + " live draws"), rinshan = new Pit(variant + " rinshan draws");
    Pit exhausted = new Pit(variant + " draws after an exhausted stream");
    Pit reveals = new Pit(variant + " dora reveals"), uras = new Pit(variant + " ura");
    int seats = variant.seats(),
        length = variant.tiles().size() - Variant.DEAD_WALL - Variant.HAND * seats;
    int[] pool = new int[TILES];
    int[] seen = {0};
    int[] draws = new int[seats];
    LedgerGames.walk(
        variant,
        104_729L + variant.ordinal(),
        LedgerGames.CALLS,
        rounds,
        new LedgerGames.Visitor() {
          @Override
          public void decision(Engine engine, RuleWorkspace ws) {
            read(engine.full());
          }

          @Override
          public void roundEnd(Engine engine) {
            read(engine.full());
            seen[0] = 0;
            Arrays.fill(draws, 0);
          }

          private void read(FullView full) {
            ChanceLog log = full.chance();
            int dealer = full.seat(0).dealer();
            for (int index = seen[0]; index < log.size(); index++) {
              LedgerGames.poolBefore(log, index, variant, pool);
              int kind = Tiles.kind(log.tile(index));
              double v = pitRandom.nextDouble();
              switch (log.kind(index)) {
                case DRAW, RINSHAN -> {
                  (log.kind(index) == ChanceKind.DRAW ? live : rinshan).add(pool, kind, v);
                  int seat = log.seat(index), offset = Math.floorMod(seat - dealer, seats);
                  if (++draws[seat] > (length - offset + seats - 1) / seats)
                    exhausted.add(pool, kind, v);
                }
                case INDICATOR -> reveals.add(pool, kind, v);
                case URA -> uras.add(pool, kind, v);
              }
            }
            seen[0] = log.size();
          }
        });
    live.check(30L * rounds);
    rinshan.check(100);
    reveals.check(100);
    uras.check(100);
    exhausted.check(1);
  }

  /** 牌譜の局の判断の時点を 1 つずつ受け取る。 */
  interface Probe {
    /**
     * @param replay 牌譜の局（判断の時点で止まっている）
     * @param line この時点までを観測したトラッカー。どこまで読んだかを持ち、帳簿を書く先の席は持たない
     * @param choices この判断の時点で選んだ手
     * @param followSeed この後の手を選ぶ乱数の種
     * @return 探し終えたら true
     */
    boolean visit(WallReplay replay, LedgerTracker line, int[] choices, long followSeed);
  }

  /** ランダムな卓と牌山で牌譜の局を 1 つ打ち、判断の時点ごとに {@code probe} を呼ぶ。{@code probe} が探し終えたら true を返す。 */
  static boolean line(Variant variant, long seed, LedgerGames.Style style, Probe probe) {
    return line(variant, seed, style, LedgerGames.RECORDING, null, probe);
  }

  /**
   * {@link #line(Variant, long, LedgerGames.Style, Probe)} と同じ局を、指定した帳簿のトラッカーで打つ。{@code fixed} が null でなければ、
   * ランダムな卓の代わりにその卓から打つ。
   */
  static boolean line(
      Variant variant,
      long seed,
      LedgerGames.Style style,
      LedgerRecording recording,
      RoundStart fixed,
      Probe probe) {
    var random = new Random(seed);
    TileSet set = variant.tiles();
    int[] scores = new int[variant.seats()];
    Arrays.fill(scores, variant.startPoints());
    RoundStart start =
        fixed != null
            ? fixed
            : new RoundStart(
                variant, random.nextInt(variant.finalRound() + 1), random.nextInt(3), 0, scores);
    int[] wall = new int[set.size()];
    for (int index = 0; index < wall.length; index++) wall[index] = set.physicalId(index);
    for (int index = wall.length - 1; index > 0; index--) {
      int other = random.nextInt(index + 1), tile = wall[index];
      wall[index] = wall[other];
      wall[other] = tile;
    }
    var replay = new WallReplay(start, wall);
    var line = new LedgerTracker(recording, LedgerGames.stakes(recording, variant));
    var ws = new RuleWorkspace(variant);
    RandomGames.Chooser chooser = LedgerGames.chooser(variant, style);
    for (int turn = 0; replay.boundary() == Boundary.DECISIONS; turn++) {
      Engine engine = replay.engine();
      line.observe(engine.full(), engine.handStart(), null, List.of(), ws);
      long choiceSeed = SeedMixer.mix64(seed * 1_000_003L + turn);
      var choiceRandom = new Random(choiceSeed);
      Decisions decisions = engine.decisions();
      int[] choices = new int[decisions.size()];
      for (int slot = 0; slot < choices.length; slot++)
        choices[slot] = chooser.choose(engine, slot, choiceRandom);
      if (probe.visit(replay, line, choices, SeedMixer.mix64(choiceSeed + 1))) return true;
      replay.commit(choices);
    }
    return false;
  }

  /**
   * 写しの局を {@code choices} の手で進め、その後は {@code chooser} が選ぶ手で進める。{@code until} が真になるか局が終わると止まり、
   * そのときの {@code Boundary} を返す。
   */
  static Boundary follow(
      Engine engine,
      int[] choices,
      RandomGames.Chooser chooser,
      long followSeed,
      RuleWorkspace ws,
      Predicate<Engine> until) {
    var random = new Random(followSeed);
    Boundary boundary = engine.commit(choices, ws);
    int[] next = new int[engine.full().variant().seats()];
    while (boundary == Boundary.DECISIONS && !until.test(engine)) {
      Decisions decisions = engine.decisions();
      for (int slot = 0; slot < decisions.size(); slot++)
        next[slot] = chooser.choose(engine, slot, random);
      boundary = engine.commit(next, ws);
    }
    return boundary;
  }

  /** ツモの直後の手番かを返す。判断する席が 1 つだけで、その席の最後のイベントが自分のツモのときに true。 */
  static boolean freshDraw(Engine engine) {
    Decisions decisions = engine.decisions();
    if (decisions.size() != 1 || decisions.kind(0) != EngineDecisionKind.TURN) return false;
    TurnEvent last = engine.full().seat(decisions.seat(0)).last();
    return last.kind() == TurnEvent.Kind.DRAW && last.seat() == decisions.seat(0);
  }

  static boolean legal(Decisions decisions, int slot, Action.Type type) {
    for (int a = 0; a < decisions.legalCount(slot); a++)
      if (LedgerGames.type(decisions, slot, a) == type) return true;
    return false;
  }

  /**
   * 牌譜の局から、ツモの直後の手番をいくつか選ぶ（暗槓を選べる手番を含む）。それぞれで、読んでいない位置を並べ替えた写しで同じ手を打ち、
   * 次の偶然のイベントで出た牌の種類を数える。期待値は判断の時点のプール（{@link LedgerTables#pool}）での n_t / N。これで、王牌・裏ドラ・
   * まだめくっていない表示牌・嶺上・赤の数え方と、プールの式を確かめる。
   */
  static void poolMatches(Variant variant, int plainWanted, int ankanWanted, int replicates) {
    int[] found = new int[2];
    var ws = new RuleWorkspace(variant);
    LedgerTables tables = LedgerTables.of(variant);
    RandomGames.Chooser chooser = LedgerGames.chooser(variant, LedgerGames.KANS);
    int[] tested = {0};
    for (long seed = 1; seed < 4000 && (found[0] < plainWanted || found[1] < ankanWanted); seed++) {
      long lineSeed = SeedMixer.mix64(seed * 7919 + variant.ordinal());
      int[] turns = {0};
      line(
          variant,
          lineSeed,
          LedgerGames.KANS,
          (replay, line, choices, followSeed) -> {
            Engine engine = replay.engine();
            if (!freshDraw(engine)) return false;
            boolean ankan = legal(engine.decisions(), 0, Action.Type.ANKAN);
            int kind = ankan ? 1 : 0;
            if (ankan ? found[1] >= ankanWanted : found[0] >= plainWanted || ++turns[0] % 11 != 5)
              return false;
            found[kind]++;
            int[] pool = new int[TILES];
            int total = tables.pool(engine.full(), pool);
            int position = engine.full().chance().size();
            long[] observed = new long[TILES];
            int counted = 0;
            for (int r = 0; r < replicates; r++) {
              Engine copy = replay.replay(SeedMixer.mix64(lineSeed * 31 + r));
              follow(
                  copy, choices, chooser, followSeed, ws, e -> e.full().chance().size() > position);
              ChanceLog log = copy.full().chance();
              if (log.size() == position) continue;
              observed[Tiles.kind(log.tile(position))]++;
              counted++;
            }
            if (counted < replicates / 2) return false;
            double[] expected = new double[TILES];
            for (int t = 0; t < TILES; t++) expected[t] = counted * (double) pool[t] / total;
            double statistic = LedgerGames.chiSquare(observed, expected);
            double p = LedgerGames.chiSquareUpper(statistic, LedgerGames.degrees(expected));
            Assert.assertTrue(
                p > 1e-4,
                variant
                    + " line "
                    + lineSeed
                    + " (ankan "
                    + ankan
                    + ") chi2="
                    + statistic
                    + " p="
                    + p);
            tested[0]++;
            return found[0] >= plainWanted && found[1] >= ankanWanted;
          });
    }
    Assert.assertTrue(found[1] >= 1, variant + " no boundary with a concealed kan");
    Assert.assertTrue(
        tested[0] >= (plainWanted + ankanWanted + 1) / 2,
        variant + " tested " + tested[0] + " " + Arrays.toString(found));
  }

  /**
   * ツモの直後の手番を分ける。0 = 聴牌の他家がいる、1 = 立直の他家がいる、2 = 槓ドラの表示を待っている（嶺上のツモの後で、打牌の後に
   * めくる）、3 = 暗槓を選べる。いくつも当てはまるときは 2、3、1、0 の順に先に当てはまったものを返す。どれでもなければ −1。
   */
  static int category(Engine engine) {
    if (!freshDraw(engine)) return -1;
    FullView full = engine.full();
    int seat = engine.decisions().seat(0);
    if (full.seat(seat).last().doraPending()) return 2;
    if (legal(engine.decisions(), 0, Action.Type.ANKAN)) return 3;
    for (int other = 0; other < full.variant().seats(); other++)
      if (other != seat && full.seat(other).riichi(other) != RiichiState.NONE) return 1;
    for (int other = 0; other < full.variant().seats(); other++)
      if (other != seat && full.waits(other) != 0) return 0;
    return -1;
  }

  /**
   * 帳簿の増分の条件付きの平均が 0 になることを確かめる。牌譜の局の判断の時点をいくつか固定する（{@link #category} の 4 種類の時点を
   * 含む）。それぞれで、読んでいない位置だけを並べ替えた写しで同じ手を打ち、次にイベントが起きた判断の時点までの帳簿の増分を、席と列ごとに
   * 平均する。並べ替えは、イベントの前に分かっていることを変えずに、見えない牌だけを引き直す。そのため c がツモの後の手・裏ドラ・引いた牌の
   * 赤などを読んでいれば、平均がずれる。時点ごとに |z| &lt; 4.5 を求める。点の列の単位が Φ の帳簿でも同じ。
   */
  static void conditionalMean(
      Variant variant, LedgerRecording recording, int[] wanted, int replicates) {
    int seats = variant.seats(), columns = LedgerColumn.COUNT;
    int[] found = new int[wanted.length];
    int[] tested = {0};
    var ws = new RuleWorkspace(variant);
    RandomGames.Chooser chooser = LedgerGames.chooser(variant, LedgerGames.KANS);
    for (long seed = 1; seed < 4000 && !Arrays.equals(found, wanted); seed++) {
      long lineSeed = SeedMixer.mix64(seed * 15_485_863L + variant.ordinal());
      line(
          variant,
          lineSeed,
          LedgerGames.KANS,
          recording,
          null,
          (replay, line, choices, followSeed) -> {
            int category = category(replay.engine());
            if (category < 0 || found[category] >= wanted[category]) return false;
            found[category]++;
            double[][] sum = new double[seats][columns], square = new double[seats][columns];
            int counted = 0;
            for (int r = 0; r < replicates; r++) {
              Engine copy = replay.replay(SeedMixer.mix64(lineSeed * 998_244_353L + r));
              var tracker = new LedgerTracker(line);
              List<LedgerTracker.Event> events = new ArrayList<>();
              tracker.trace(events);
              List<LedgerGames.Seat> targets = LedgerGames.seats(variant);
              follow(
                  copy,
                  choices,
                  chooser,
                  followSeed,
                  ws,
                  e -> {
                    tracker.observe(e.full(), e.handStart(), null, targets, ws);
                    return !events.isEmpty();
                  });
              if (events.isEmpty()) continue;
              counted++;
              for (LedgerGames.Seat target : targets)
                for (int f = 0; f < columns; f++) {
                  double value = target.ledger()[f];
                  sum[target.seat()][f] += value;
                  square[target.seat()][f] += value * value;
                }
            }
            if (counted < replicates / 2) return false;
            tested[0]++;
            for (int seat = 0; seat < seats; seat++)
              for (int f = 0; f < columns; f++) {
                String where =
                    variant
                        + " "
                        + recording.pointUnit()
                        + " line "
                        + lineSeed
                        + " category "
                        + category
                        + " seat "
                        + seat
                        + " "
                        + COLUMNS[f].reportName();
                double mean = sum[seat][f] / counted,
                    variance = square[seat][f] / counted - mean * mean;
                if (variance <= 1e-18) {
                  Assert.assertEquals(mean, 0, 1e-9, where);
                  continue;
                }
                double z = mean / Math.sqrt(variance / counted);
                Assert.assertTrue(Math.abs(z) < 4.5, where + " mean=" + mean + " z=" + z);
              }
            return Arrays.equals(found, wanted);
          });
    }
    int all = Arrays.stream(wanted).sum();
    Assert.assertTrue(
        tested[0] >= (all + 1) / 2,
        variant + " tested " + tested[0] + " " + Arrays.toString(found));
  }

  /**
   * 順位が動かない卓で、点の列を Φ の単位で付けた帳簿が恒等的に 0 になることを確かめる。最後の局を、親が大差の首位の卓から {@code lines} 局
   * 打つ。どの和了でも流局でも半荘が終わり、親は首位のままなので、親の席の点の列は全ての局でちょうど 0。同じ局を千点で付けた帳簿では、
   * 親の席の点の列が 0 でない局がある。順位が動く子の席では、Φ の単位でも 0 でない局がある。
   */
  static void decidedTable(Variant variant, int lines) {
    int seats = variant.seats(), trailing = 5000;
    int[] scores = new int[seats];
    Arrays.fill(scores, variant.startPoints());
    var table = new TableState(variant);
    table.set(variant.finalRound(), 0, 0, scores);
    int dealer = table.dealer();
    Arrays.fill(scores, trailing);
    scores[dealer] = seats * variant.startPoints() - (seats - 1) * trailing;
    var start = new RoundStart(variant, variant.finalRound(), 0, 0, scores);
    var ws = new RuleWorkspace(variant);
    RandomGames.Chooser chooser = LedgerGames.chooser(variant, LedgerGames.PLAIN);
    // [単位][親か子か] の、点の列が 0 でなかった局の数
    int[][] nonzero = new int[2][2];
    List<LedgerRecording> recordings = List.of(LedgerGames.RECORDING, LedgerGames.PHI_RECORDING);
    for (long seed = 1; seed <= lines; seed++) {
      long lineSeed = SeedMixer.mix64(seed * 86_028_121L + variant.ordinal());
      for (int unit = 0; unit < recordings.size(); unit++) {
        int u = unit;
        line(
            variant,
            lineSeed,
            LedgerGames.PLAIN,
            recordings.get(unit),
            start,
            (replay, line, choices, followSeed) -> {
              Engine copy = replay.engine().copy();
              var tracker = new LedgerTracker(line);
              List<LedgerGames.Seat> targets = LedgerGames.seats(variant);
              follow(
                  copy,
                  choices,
                  chooser,
                  followSeed,
                  ws,
                  e -> {
                    tracker.observe(e.full(), e.handStart(), null, targets, ws);
                    return false;
                  });
              boolean[] moved = new boolean[2];
              for (LedgerGames.Seat target : targets)
                for (LedgerColumn column : COLUMNS)
                  if (column.group() == LedgerGroup.POINTS
                      && target.ledger()[column.ordinal()] != 0)
                    moved[target.seat() == dealer ? 0 : 1] = true;
              for (int side = 0; side < 2; side++) if (moved[side]) nonzero[u][side]++;
              return true;
            });
      }
    }
    Assert.assertEquals(nonzero[1][0], 0, variant + " PHI point columns of the leading dealer");
    Assert.assertTrue(nonzero[0][0] > lines / 4, variant + " THOUSANDS dealer " + nonzero[0][0]);
    Assert.assertTrue(nonzero[1][1] > lines / 4, variant + " PHI others " + nonzero[1][1]);
  }

  /**
   * 局末の裏ドラの項の条件付きの平均が 0 になることを確かめる。立直した席が和了を選ぶ判断の時点を、ロンとツモで {@code wins} 局ずつ
   * 探す。それぞれで、裏ドラを含む読んでいない位置を並べ替えた写しで同じ手を打ち、局末に {@link LedgerTracker#handEnd} が書いた
   * 裏ドラの列を席ごとに平均する。和了の手と重みは並べ替えで変わらないので、局ごと・席ごとに |z| &lt; 4.5 を求める。ほかの列は 0 で、
   * 和了者の裏ドラの列は、半分より多くの写しで 0 でない（和了の手が数えられている）。
   */
  static void uraCentred(Variant variant, int wins, int replicates) {
    int seats = variant.seats();
    int ura = LedgerColumn.URA.ordinal();
    int[] found = new int[2];
    var ws = new RuleWorkspace(variant);
    for (long seed = 1; seed < 8000 && (found[0] < wins || found[1] < wins); seed++) {
      long lineSeed = SeedMixer.mix64(seed * 32_452_843L + variant.ordinal());
      line(
          variant,
          lineSeed,
          LedgerGames.PLAIN,
          (replay, line, choices, followSeed) -> {
            Engine engine = replay.engine();
            Decisions decisions = engine.decisions();
            int winner = -1;
            boolean ron = false;
            for (int slot = 0; slot < decisions.size(); slot++) {
              Action.Type type = LedgerGames.type(decisions, slot, choices[slot]);
              int seat = decisions.seat(slot);
              if ((type == Action.Type.RON_AGARI || type == Action.Type.TSUMO_AGARI)
                  && engine.full().seat(seat).riichi(seat) != RiichiState.NONE) {
                winner = seat;
                ron = type == Action.Type.RON_AGARI;
              }
            }
            int kind = ron ? 0 : 1;
            if (winner < 0 || found[kind] >= wins) return false;
            double[] sum = new double[seats], square = new double[seats];
            int nonzero = 0;
            for (int r = 0; r < replicates; r++) {
              Engine copy = replay.replay(SeedMixer.mix64((lineSeed + 13) * 1_000_000_009L + r));
              var tracker = new LedgerTracker(line);
              Assert.assertNotEquals(copy.commit(choices, ws), Boundary.DECISIONS);
              RoundEnd end = copy.roundEnd();
              RoundOutcome outcome = end.outcome();
              if (!outcome.kind().win()) return false;
              List<LedgerGames.Seat> targets = LedgerGames.seats(variant);
              tracker.handEnd(copy.full(), end, targets);
              for (LedgerGames.Seat target : targets) {
                int seat = target.seat();
                double value = target.ledger()[ura];
                for (int f = 0; f < LedgerColumn.COUNT; f++)
                  if (f != ura)
                    Assert.assertEquals(
                        target.ledger()[f],
                        0.0,
                        variant + " seat " + seat + " " + COLUMNS[f].reportName());
                sum[seat] += value;
                square[seat] += value * value;
                if (seat == winner && value != 0) nonzero++;
              }
            }
            found[kind]++;
            Assert.assertTrue(
                nonzero > replicates / 2,
                variant + " line " + lineSeed + " (ron " + ron + ") winner ura nonzero " + nonzero);
            for (int seat = 0; seat < seats; seat++) {
              double mean = sum[seat] / replicates,
                  variance = square[seat] / replicates - mean * mean;
              String where = variant + " line " + lineSeed + " (ron " + ron + ") seat " + seat;
              if (variance <= 1e-18) {
                Assert.assertEquals(mean, 0, 1e-9, where);
                continue;
              }
              double z = mean / Math.sqrt(variance / replicates);
              Assert.assertTrue(Math.abs(z) < 4.5, where + " mean=" + mean + " z=" + z);
            }
            return found[0] >= wins && found[1] >= wins;
          });
    }
    Assert.assertEquals(found[0], wins, variant + " riichi ron wins");
    Assert.assertEquals(found[1], wins, variant + " riichi tsumo wins");
  }
}
