package com.epsilon.prism.luck;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.play.CrnKeys;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.Arrays;
import java.util.List;

/**
 * 1 本の経路について、運の精算の帳簿を計算する。増分の式と平均 0 の条件はパッケージの説明に書く。受け取り先（{@link LedgerTarget}）の
 * 帳簿に、判断の時点ごとに前の判断の時点からのツモと槓ドラの表示の増分を、強制でない判断ごとに方策の偶然の増分を、局末に裏ドラの増分を足す。
 * 点の払いの式も持つ。計算するのは収集の設定（{@link LedgerRecording}）が選んだ組の列だけで、ほかの列は 0 のまま。どの列の値も、ほかの組を
 * 計算するかどうかに依らない。
 *
 * <p>1 度に 1 スレッドだけが触る。席ごとの栄和の表は stamp が同じなら使い回し、写しと分岐は親の表の配列を写して始める。表そのものは作った
 * 後に変えないので、分岐が進んで作った表が親に混ざることはない。表は経路の過去の状態だけで決まるので、ハッシュが衝突して古い表を使っても
 * 平均 0 は崩れず、結果は実行順やワーカー数に依らない。
 */
public final class LedgerTracker {

  private static final int COLUMNS = LedgerColumn.COUNT;

  private static final LedgerColumn[] BY_ORDINAL = LedgerColumn.values();

  /** 自分の受け入れの列（一向聴・二向聴・三向聴以上の順）。 */
  private static final int[] OWN_IMPROVE = {
    LedgerColumn.OWN_IMPROVE_1.ordinal(), LedgerColumn.OWN_IMPROVE_2.ordinal(), LedgerColumn.OWN_IMPROVE_3.ordinal()
  };

  private static final int OWN_TSUMO = LedgerColumn.OWN_TSUMO.ordinal(),
      OWN_FORCED_DEAL_IN = LedgerColumn.OWN_FORCED_DEAL_IN.ordinal(),
      OWN_DRAWN_DANGER = LedgerColumn.OWN_DRAWN_DANGER.ordinal(),
      OPP_IMPROVE_1 = LedgerColumn.OPP_IMPROVE_1.ordinal(),
      OPP_IMPROVE_2 = LedgerColumn.OPP_IMPROVE_2.ordinal(),
      OPP_TSUMO = LedgerColumn.OPP_TSUMO.ordinal(),
      OPP_FORCED_TO_SELF = LedgerColumn.OPP_FORCED_TO_SELF.ordinal(),
      OPP_FORCED_TO_OTHER = LedgerColumn.OPP_FORCED_TO_OTHER.ordinal(),
      OPP_DRAWN_CHANCE = LedgerColumn.OPP_DRAWN_CHANCE.ordinal(),
      DORA_SELF = LedgerColumn.DORA_SELF.ordinal(),
      DORA_OPPONENTS = LedgerColumn.DORA_OPPONENTS.ordinal(),
      URA = LedgerColumn.URA.ordinal(),
      POLICY_SELF = LedgerColumn.POLICY_SELF.ordinal(),
      POLICY_OPPONENT = LedgerColumn.POLICY_OPPONENT.ordinal();

  /** 全てのルールのうち、最も多い席の数。 */
  private static final int MAX_SEATS = Arrays.stream(Variant.values()).mapToInt(Variant::seats).max().orElseThrow();

  /** 栄和の表を作るときに、打たれたとみなす牌の写しの番号（赤でない写し）。 */
  private static final int PLAIN_COPY = Tiles.RED_COPY + 1;

  /** 手牌の枚数を long 1 つに詰めるときの種類の数（1 種類 3 bit）。 */
  private static final int PACKED_KINDS = Long.SIZE / 3;

  /** ツモの払いの割合を求めるときの基本点。100 点の倍数なので、払いを 100 点単位に切り上げても割合は変わらない。 */
  private static final int SHARE_BASE = 100;

  /** 進行ワーカーのスレッドごと・ルールごとの作業領域（採点の結果と手牌の写しがスレッドセーフでないため）。 */
  private static final ThreadLocal<Workspace[]> WORKSPACES =
      ThreadLocal.withInitial(() -> new Workspace[Variant.values().length]);

  /** 席ごとの栄和の表。 */
  private final SeatTable[] seatTables;
  private final ChanceBatch batch;

  /** 計算する組のビット集合（組の番号の位置にビットを立てる）。写しと分岐は親と同じ値を持つ。 */
  private final int groups;

  /** 点の列を切る幅（千点）と、方策の列で s を切る幅（効用）。 */
  private final double pointClip, policyClip;

  /** 計測した時間と呼び出しの数（{@link #flush} でまとめて足す）。 */
  private long nanos, calls;

  private List<Event> trace;

  /**
   * 主経路の計算器を作る。局の始めから読む。
   *
   * @param recording 計算する組と切る幅
   */
  public LedgerTracker(LedgerRecording recording) {
    int bits = 0;
    for (LedgerGroup group : recording.compute()) bits |= 1 << group.ordinal();
    groups = bits;
    pointClip = recording.pointClip();
    policyClip = recording.policyClip();
    seatTables = new SeatTable[MAX_SEATS];
    batch = new ChanceBatch();
  }

  /**
   * 写しか分岐の計算器を作る。親が最後に読んだ判断の時点から読み、親と同じ組を同じ幅で計算する。
   *
   * @param parent 親の経路の計算器
   */
  public LedgerTracker(LedgerTracker parent) {
    groups = parent.groups;
    pointClip = parent.pointClip;
    policyClip = parent.policyClip;
    seatTables = parent.seatTables.clone();
    batch = new ChanceBatch(parent.batch);
  }

  private boolean computes(LedgerGroup group) {
    return (groups & (1 << group.ordinal())) != 0;
  }

  /** 読んだイベントを、渡したリストに足すようにする（テスト専用。本番では null のまま）。 */
  void trace(List<Event> events) {
    trace = events;
  }

  /** 主経路を次の局へ移す。読む位置を局の始めに戻し、席の表を捨てる。 */
  public void reset() {
    batch.reset();
    Arrays.fill(seatTables, null);
  }

  /**
   * 計測した時間と呼び出しの数を指定した段に足し、計測を 0 に戻す（進行の終わりに 1 回呼ぶ）。呼び出しが無ければ何もしない。
   *
   * @param timer 足す先
   * @param stage 足す段
   */
  public <S extends Enum<S>> void flush(StageTimer<S> timer, S stage) {
    if (calls == 0) return;
    timer.add(stage, nanos, calls);
    nanos = 0;
    calls = 0;
  }

  /**
   * 判断の時点ごとに 1 回呼ぶ（強制の判断も含む）。前の判断の時点からのイベントの束（槓ドラの表示が古い順に並び、ツモが最後）を、プールを
   * イベントの前に戻して計算する。帳簿を持つ受け取り先の席ごとに増分を求め、その席の全ての受け取り先の帳簿に足す。
   *
   * @param full 判断の時点の局面
   * @param targets 経路の受け取り先
   * @param ws 採点に使うルールの作業領域
   * @throws IllegalStateException 束の並びが崩れているとき、プールが空のとき、帳簿の値が有限でないとき
   */
  public void observe(FullView full, List<? extends LedgerTarget> targets, RuleWorkspace ws) {
    ChanceLog log = full.chance();
    batch.read(log);
    boolean dora = computes(LedgerGroup.DORA);
    boolean draws = computes(LedgerGroup.IMPROVE) || computes(LedgerGroup.POINTS);
    int seats = seats(targets);
    if (seats == 0 || batch.size() == 0 || !(dora || draws)) return;
    long started = System.nanoTime();
    Workspace work = workspace(full.variant());
    int draw = batch.draw(), reveals = batch.reveals();
    int drawer = draw < 0 ? -1 : log.seat(draw);
    int drawn = draw < 0 ? -1 : Tiles.kind(log.tile(draw));
    int[] pool = work.pool;
    work.tables.pool(full, pool);
    // 束のイベントを全て戻して、束の最初のイベントの前のプールにする
    for (int k = 0; k < reveals; k++) pool[Tiles.kind(log.tile(batch.reveal(k)))]++;
    if (draw >= 0) pool[drawn]++;
    double[][] delta = work.delta;
    for (double[] row : delta) Arrays.fill(row, 0);
    if (reveals > 0) {
      if (dora) owned(full, work, drawer, drawn);
      for (int k = 0; k < reveals; k++) {
        int index = batch.reveal(k);
        int indicator = Tiles.kind(log.tile(index));
        if (trace != null) trace.add(event(full, work, log, index, pool, drawer, drawn));
        if (dora) reveal(work, pool, indicator, seats, delta);
        pool[indicator]--;
      }
    }
    if (draw >= 0) {
      if (trace != null) trace.add(event(full, work, log, draw, pool, drawer, drawn));
      if (draws) draw(full, ws, work, drawer, drawn, pool, seats, delta);
    }
    apply(delta, targets, 0, URA);
    nanos += System.nanoTime() - started;
    calls++;
  }

  /**
   * 判断の後の強制でない判断ごとに 1 回呼ぶ（手を選んだ後、記録を作る前）。s が無い判断と、b(a) &gt; 0 の手に有限でない s がある判断は
   * 足さない（{@link #policyCounts}）。受け取り先の席が判断した席なら自分の方策の列に、ほかの席なら他家の方策の列に足す。
   *
   * @param decider 判断した席
   * @param scores 手ごとの s（無ければ null）
   * @param p {@link CrnKeys#select} に渡した分布
   * @param chosen 選んだ手の位置
   * @param n 手の数
   * @param targets 経路の受け取り先
   */
  public void policy(int decider, float[] scores, float[] p, int chosen, int n, List<? extends LedgerTarget> targets) {
    if (scores == null || !computes(LedgerGroup.POLICY)) return;
    if (seats(targets) == 0) return;
    long started = System.nanoTime();
    if (policyCounts(scores, p, n)) {
      double value = policyDelta(scores, p, chosen, n, policyClip);
      for (LedgerTarget target : targets) {
        double[] ledger = target.ledger();
        int seat = target.seat();
        if (ledger == null || seat < 0) continue;
        add(ledger, seat == decider ? POLICY_SELF : POLICY_OPPONENT, value);
      }
    }
    nanos += System.nanoTime() - started;
    calls++;
  }

  /**
   * 局末に 1 回呼ぶ（次の局を始める前）。立直して和了した席 w ごとに裏ドラの増分を求め、席 i から見た重みを掛けて裏ドラの列に足す。
   * 表示牌 t が指すドラを和了形が何枚持つかを g_w(t) = own_w(ドラ(t)) とすると、増分は実際の値 Σ g_w(裏) から期待値
   * k·Σ_t n_t·g_w(t)/N を引いたもの（k は裏ドラの表示の数、n は裏ドラをめくる前のプール）。重みは、i = w なら +1。栄和なら放銃した席が
   * −1 で、ほかの席は 0。ツモなら i の払いの割合の負（四人打ちの子の和了なら親 −1/2・子 −1/4、親の和了なら −1/3）。
   *
   * @param full 局末の局面
   * @param end 局の終わり方
   * @param targets 経路の受け取り先
   * @throws IllegalStateException 最後に読んだ判断の時点の後にツモ（嶺上ツモを含む）があるとき。裏ドラの表示牌の後に槓ドラの表示牌が
   *     あるとき
   */
  public void handEnd(FullView full, RoundEnd end, List<? extends LedgerTarget> targets) {
    ChanceLog log = full.chance();
    int size = log.size(), uras = 0;
    for (int index = batch.position(); index < size; index++) {
      switch (log.kind(index)) {
        case INDICATOR -> {
          if (uras > 0) throw new IllegalStateException("Indicator " + index + " follows the ura indicators");
        }
        case URA -> uras++;
        case DRAW, RINSHAN ->
            throw new IllegalStateException("Draw " + index + " after the last decision of the round");
      }
    }
    RoundOutcome outcome = end.outcome();
    if (uras == 0 || !outcome.kind().win() || !computes(LedgerGroup.URA)) return;
    int seats = seats(targets);
    if (seats == 0) return;
    long started = System.nanoTime();
    Workspace work = workspace(full.variant());
    LedgerTables tables = work.tables;
    int[] pool = work.pool;
    int total = tables.pool(full, pool);
    int first = size - uras;
    // 裏ドラをめくる前のプール
    for (int index = first; index < size; index++) pool[Tiles.kind(log.tile(index))]++;
    total += uras;
    int dealer = full.seat(0).dealer();
    double[][] delta = work.delta;
    for (double[] row : delta) Arrays.fill(row, 0);
    int[] counts = work.owned[0];
    int[] doraOf = tables.doraOf;
    for (int i = 0; i < outcome.winnerCount(); i++) {
      int winner = outcome.winner(i);
      if (full.seat(winner).riichi(winner) == RiichiState.NONE) continue;
      tables.winningHand(full, winner, work.scratch, counts);
      double realized = 0, expected = 0;
      for (int index = first; index < size; index++) realized += counts[doraOf[Tiles.kind(log.tile(index))]];
      for (long bits = tables.kinds; bits != 0; bits &= bits - 1) {
        int kind = Long.numberOfTrailingZeros(bits);
        expected += pool[kind] * counts[doraOf[kind]];
      }
      double term = realized - uras * expected / total;
      for (int seat = 0; seat < tables.seats; seat++)
        if ((seats & (1 << seat)) != 0)
          add(delta[seat], URA, uraWeight(tables.variant, outcome, winner, seat, dealer) * term);
    }
    apply(delta, targets, URA, URA + 1);
    nanos += System.nanoTime() - started;
    calls++;
  }

  /**
   * ツモ 1 回分の、受け入れの列と点の列の増分を、受け取り先の席ごとに足す。{@code drawer} はツモった席、{@code drawn} は引いた種類、
   * {@code pool} はツモの前のプール。
   */
  private void draw(
      FullView full,
      RuleWorkspace ws,
      Workspace work,
      int drawer,
      int drawn,
      int[] pool,
      int seats,
      double[][] delta) {
    LedgerTables tables = work.tables;
    int total = total(tables, pool);
    if (total <= 0) throw new IllegalStateException("Empty pool before a draw");
    HandShape before = work.before;
    full.copyHand(drawer, before);
    before.remove(drawn);
    int shanten = work.shapes.shanten(before);
    long accepting = work.shapes.improving(before);
    boolean improve = computes(LedgerGroup.IMPROVE), points = computes(LedgerGroup.POINTS);
    // ツモの和了（聴牌のときだけ）: 引いた牌を同じ種類の赤でない牌に替えて採点する。プールにない種類は引けず、増分に効かないので
    // 採点しない
    long tsumo = 0;
    if (points && shanten == 0) {
      for (long bits = accepting; bits != 0; bits &= bits - 1) {
        int tile = Long.numberOfTrailingZeros(bits);
        if (pool[tile] == 0) continue;
        if (!full.tsumoScore(drawer, tile, ws, work.score)) continue;
        tsumo |= 1L << tile;
        work.tsumoBase[tile] = work.score.base();
      }
    }
    Variant variant = tables.variant;
    int count = tables.seats;
    SeatView view = full.seat(drawer);
    boolean riichi = view.riichi(drawer) != RiichiState.NONE;
    int dealer = view.dealer(), honba = view.honba(), sticks = view.kyotaku();
    // 栄和の表を作る席: ツモった席が受け取り先なら（自分の放銃）全ての他家。ツモった席が立直でも（ほかの席への放銃）全ての他家。ほかは受け取り先の席だけ
    boolean everyone = (seats & (1 << drawer)) != 0 || riichi;
    SeatTable[] seen = work.seen;
    for (int seat = 0; seat < count; seat++)
      seen[seat] =
          points && seat != drawer && (everyone || (seats & (1 << seat)) != 0) ? table(full, ws, work, seat) : null;
    double[] values = work.values;
    for (int seat = 0; seat < count; seat++) {
      if ((seats & (1 << seat)) == 0) continue;
      double[] row = delta[seat];
      if (seat == drawer) {
        if (improve && shanten >= 1)
          add(row, OWN_IMPROVE[Math.min(shanten, OWN_IMPROVE.length) - 1], indicator(accepting, drawn, pool, total));
        if (!points) continue;
        if (tsumo != 0) {
          for (long bits = tsumo; bits != 0; bits &= bits - 1) {
            int tile = Long.numberOfTrailingZeros(bits);
            values[tile] =
                thousands(tsumoGain(variant, work.tsumoBase[tile], drawer, dealer, honba, sticks), pointClip);
          }
          add(row, OWN_TSUMO, centred(tsumo, values, drawn, pool, total));
        }
        // 打てば他家に栄和される種類（ツモで和了できる種類は除く）
        long ron = 0;
        for (int other = 0; other < count; other++) if (other != drawer) ron |= seen[other].ron();
        ron &= ~tsumo;
        if (ron != 0) {
          for (long bits = ron; bits != 0; bits &= bits - 1) {
            int tile = Long.numberOfTrailingZeros(bits);
            double loss = 0;
            for (int other = 0; other < count; other++)
              if (other != drawer && (seen[other].ron() & (1L << tile)) != 0)
                loss += thousands(ronPayment(variant, seen[other].ronBase()[tile], other, dealer, honba), pointClip);
            values[tile] = -loss;
          }
          add(row, riichi ? OWN_FORCED_DEAL_IN : OWN_DRAWN_DANGER, centred(ron, values, drawn, pool, total));
        }
      } else {
        if (improve && shanten >= 1)
          add(row, shanten == 1 ? OPP_IMPROVE_1 : OPP_IMPROVE_2, indicator(accepting, drawn, pool, total));
        if (!points) continue;
        if (tsumo != 0) {
          for (long bits = tsumo; bits != 0; bits &= bits - 1) {
            int tile = Long.numberOfTrailingZeros(bits);
            values[tile] = -thousands(tsumoPayment(work.tsumoBase[tile], drawer, seat, dealer, honba), pointClip);
          }
          add(row, OPP_TSUMO, centred(tsumo, values, drawn, pool, total));
        }
        SeatTable own = seen[seat];
        long gain = own.ron() & ~tsumo;
        if (gain != 0) {
          for (long bits = gain; bits != 0; bits &= bits - 1) {
            int tile = Long.numberOfTrailingZeros(bits);
            int gained = ronPayment(variant, own.ronBase()[tile], seat, dealer, honba) + Variant.DEPOSIT * sticks;
            values[tile] = thousands(gained, pointClip);
          }
          add(row, riichi ? OPP_FORCED_TO_SELF : OPP_DRAWN_CHANCE, centred(gain, values, drawn, pool, total));
        }
        if (riichi) {
          long ron = 0;
          for (int other = 0; other < count; other++)
            if (other != drawer && other != seat) ron |= seen[other].ron();
          ron &= ~tsumo;
          if (ron != 0) {
            for (long bits = ron; bits != 0; bits &= bits - 1) {
              int tile = Long.numberOfTrailingZeros(bits);
              double paid = 0;
              for (int other = 0; other < count; other++)
                if (other != drawer && other != seat && (seen[other].ron() & (1L << tile)) != 0)
                  paid += thousands(ronPayment(variant, seen[other].ronBase()[tile], other, dealer, honba), pointClip);
              values[tile] = paid;
            }
            add(row, OPP_FORCED_TO_OTHER, centred(ron, values, drawn, pool, total));
          }
        }
      }
    }
  }

  /**
   * 席の栄和の表を返す。stamp が前の表と同じなら使い回し、違えば新しい表を作って入れ替える（前の表は変えない）。打つ席は表に効かないので、
   * その席の下家が打ったとして採点する。
   */
  private SeatTable table(FullView full, RuleWorkspace ws, Workspace work, int seat) {
    LedgerTables tables = work.tables;
    long stamp = stamp(full, work, seat);
    SeatTable table = seatTables[seat];
    if (table != null && table.stamp() == stamp) return table;
    int from = (seat + 1) % tables.seats;
    long ron = 0;
    int[] base = new int[Tiles.KINDS];
    int[] held = work.held;
    tables.holdings(full, seat, work.scratch, held);
    for (long bits = full.waits(seat); bits != 0; bits &= bits - 1) {
      int tile = Long.numberOfTrailingZeros(bits);
      // 形の上では待ちでも、その種類の牌を全て自分が持っていれば誰も打てない
      if (held[tile] == tables.copies[tile]) continue;
      if (!full.ronScore(seat, Tiles.physicalId(tile, PLAIN_COPY), from, ws, work.score)) continue;
      ron |= 1L << tile;
      base[tile] = work.score.base();
    }
    table = new SeatTable(stamp, ron, base);
    seatTables[seat] = table;
    return table;
  }

  /**
   * 席の栄和の表が依る状態のハッシュを返す。含めるのは、純手牌の枚数・面子・赤牌を持つか・抜いた北・立直（ダブル立直を含む）・一発・
   * 見逃しの振聴・河の牌の種類（捨て牌の振聴）・山が尽きたか（河底）・第一巡か・局・表示牌。
   */
  private static long stamp(FullView full, Workspace work, int seat) {
    SeatView view = full.seat(seat);
    HandShape hand = work.scratch;
    full.copyHand(seat, hand);
    long low = 0, high = 0;
    for (long bits = hand.kinds(); bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      long count = hand.count(kind);
      if (kind < PACKED_KINDS) low |= count << (3 * kind);
      else high |= count << (3 * (kind - PACKED_KINDS));
    }
    long red = 0;
    for (long bits = work.tables.redKinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      if (view.holds(Tiles.physicalId(kind, Tiles.RED_COPY))) red |= 1L << kind;
    }
    long melds = view.meldCount(seat);
    for (int m = 0; m < view.meldCount(seat); m++) melds = SeedMixer.mix64(melds ^ view.meld(seat, m));
    long dora = view.indicatorCount();
    for (int k = 0; k < view.indicatorCount(); k++) dora = (dora << 6) | Tiles.kind(view.indicator(k));
    EventLog events = view.events();
    long flags =
        view.riichi(seat).ordinal()
            | (view.ippatsu(seat) ? 4 : 0)
            | (full.furiten(seat) ? 8 : 0)
            | (view.liveLeft() == 0 ? 16 : 0)
            | (events.interruptions() == 0 && events.riverSize(seat) == 0 ? 32 : 0)
            | ((long) view.nuki(seat) << 8)
            | ((long) view.round() << 16);
    long h = SeedMixer.mix64(low);
    h = SeedMixer.mix64(h ^ high);
    h = SeedMixer.mix64(h ^ red);
    h = SeedMixer.mix64(h ^ melds);
    h = SeedMixer.mix64(h ^ flags);
    h = SeedMixer.mix64(h ^ view.riverKinds(seat));
    return SeedMixer.mix64(h ^ dora);
  }

  /**
   * 点を千点単位に直し、上下の幅で切った値を返す。
   *
   * @param points 点
   * @param clip 上限（千点。下限はその負）
   */
  static double thousands(int points, double clip) {
    return Math.max(-clip, Math.min(clip, points / 1000.0));
  }

  /**
   * s を上下の幅で切った値を返す。
   *
   * @param score s
   * @param clip 上限（効用。下限はその負）
   */
  static double clipScore(float score, double clip) {
    return Math.max(-clip, Math.min(clip, score));
  }

  /** 増分を足す。値が有限でなければ帳簿の計算の誤りなので、NaN を黙って行に入れずに例外を投げる。 */
  static void add(double[] delta, int column, double value) {
    if (!Double.isFinite(value))
      throw new IllegalStateException("Non-finite ledger term " + BY_ORDINAL[column].reportName() + " = " + value);
    delta[column] += value;
  }

  /**
   * 方策の項を足すかを返す。b(a) &gt; 0 の手の s が全て有限で、重みの和が正なら足す。b(a) &gt; 0 は {@code p > 0} のことで、
   * {@link CrnKeys#select} と同じく NaN・0・負の手は重み 0 とする。足すかどうかは手の抽選の前に決まる。
   */
  static boolean policyCounts(float[] scores, float[] p, int n) {
    double sum = 0;
    for (int a = 0; a < n; a++) {
      float probability = p[a];
      if (!(probability > 0)) continue;
      if (!Float.isFinite(scores[a])) return false;
      sum += probability;
    }
    return sum > 0;
  }

  /**
   * 方策の項 δ = s̄_A − Σ_a b(a)·s̄_a を返す（b(a) = w_a / Σ w、w_a = p_a &gt; 0 ? p_a : 0。s̄ は s を ±clip で切ったもの）。
   * {@link #policyCounts} が真のときだけ呼ぶ。
   *
   * @param clip s を切る幅（効用）
   */
  static double policyDelta(float[] scores, float[] p, int chosen, int n, double clip) {
    double sum = 0, weighted = 0;
    for (int a = 0; a < n; a++) {
      float probability = p[a];
      double weight = probability > 0 ? probability : 0;
      if (weight == 0) continue;
      sum += weight;
      weighted += weight * clipScore(scores[a], clip);
    }
    return clipScore(scores[chosen], clip) - weighted / sum;
  }

  /** 今のスレッドで、このルールに使う作業領域を返す（無ければ作る）。 */
  private static Workspace workspace(Variant variant) {
    Workspace[] all = WORKSPACES.get();
    Workspace work = all[variant.ordinal()];
    if (work == null) {
      work = new Workspace(LedgerTables.of(variant));
      all[variant.ordinal()] = work;
    }
    return work;
  }

  /** 帳簿を持つ受け取り先の席を、ビット集合にして返す。 */
  private static int seats(List<? extends LedgerTarget> targets) {
    int seats = 0;
    for (LedgerTarget target : targets)
      if (target.ledger() != null && target.seat() >= 0) seats |= 1 << target.seat();
    return seats;
  }

  /** 席ごとの増分の列 [from, to) を、その席の全ての受け取り先の帳簿に足す。 */
  private static void apply(double[][] delta, List<? extends LedgerTarget> targets, int from, int to) {
    for (LedgerTarget target : targets) {
      double[] ledger = target.ledger();
      int seat = target.seat();
      if (ledger == null || seat < 0) continue;
      double[] row = delta[seat];
      for (int column = from; column < to; column++) ledger[column] += row[column];
    }
  }

  private static int total(LedgerTables tables, int[] pool) {
    int total = 0;
    for (long bits = tables.kinds; bits != 0; bits &= bits - 1) total += pool[Long.numberOfTrailingZeros(bits)];
    return total;
  }

  /** 席ごとに、持つ牌（純手牌・面子・抜いた北）の枚数を書く。ツモった席があれば、その席からは引いた牌を外す（ツモの前の枚数にする）。 */
  private static void owned(FullView full, Workspace work, int drawer, int drawn) {
    for (int seat = 0; seat < work.tables.seats; seat++) {
      int[] counts = work.owned[seat];
      work.tables.holdings(full, seat, work.scratch, counts);
      if (seat == drawer) counts[drawn]--;
    }
  }

  /** 読んだイベントの記録を作る（テスト専用）。 */
  private static Event event(
      FullView full, Workspace work, ChanceLog log, int index, int[] pool, int drawer, int drawn) {
    int[][] hands = new int[work.tables.seats][Tiles.KINDS];
    HandShape hand = work.scratch;
    for (int seat = 0; seat < work.tables.seats; seat++) {
      full.copyHand(seat, hand);
      for (long bits = hand.kinds(); bits != 0; bits &= bits - 1) {
        int kind = Long.numberOfTrailingZeros(bits);
        hands[seat][kind] = hand.count(kind);
      }
      if (seat == drawer) hands[seat][drawn]--;
    }
    return new Event(log.kind(index), log.seat(index), log.tile(index), pool.clone(), hands);
  }

  /**
   * 槓ドラの表示 1 回分の増分を足す。{@code pool} は表示の前のプール、{@code indicator} は表示牌の種類 u。自分のドラの列には
   * own_i(ドラ(u)) の、他家のドラの列には Σ_{w≠i} own_w(ドラ(u)) の、プールの上の平均からのずれを足す。
   */
  private static void reveal(Workspace work, int[] pool, int indicator, int seats, double[][] delta) {
    LedgerTables tables = work.tables;
    int total = total(tables, pool);
    if (total <= 0) throw new IllegalStateException("Empty pool before a dora reveal");
    int[] doraOf = tables.doraOf;
    double[] deviation = work.deviation;
    int dora = doraOf[indicator];
    for (int seat = 0; seat < tables.seats; seat++) {
      int[] owned = work.owned[seat];
      double expected = 0;
      for (long bits = tables.kinds; bits != 0; bits &= bits - 1) {
        int kind = Long.numberOfTrailingZeros(bits);
        expected += pool[kind] * owned[doraOf[kind]];
      }
      deviation[seat] = owned[dora] - expected / total;
    }
    for (int seat = 0; seat < tables.seats; seat++) {
      if ((seats & (1 << seat)) == 0) continue;
      double others = 0;
      for (int other = 0; other < tables.seats; other++) if (other != seat) others += deviation[other];
      add(delta[seat], DORA_SELF, deviation[seat]);
      add(delta[seat], DORA_OPPONENTS, others);
    }
  }

  /** 指示関数の増分 [d ∈ mask] − Σ_{t∈mask} n_t / N を返す。 */
  private static double indicator(long mask, int drawn, int[] pool, int total) {
    int live = 0;
    for (long bits = mask; bits != 0; bits &= bits - 1) live += pool[Long.numberOfTrailingZeros(bits)];
    return ((mask >>> drawn) & 1) - (double) live / total;
  }

  /** mask の上では values の値、外では 0 をとる関数 c の増分 c(d) − Σ_{t∈mask} n_t·c(t) / N を返す。 */
  private static double centred(long mask, double[] values, int drawn, int[] pool, int total) {
    double expected = 0;
    for (long bits = mask; bits != 0; bits &= bits - 1) {
      int tile = Long.numberOfTrailingZeros(bits);
      expected += pool[tile] * values[tile];
    }
    double realized = ((mask >>> drawn) & 1) != 0 ? values[drawn] : 0;
    return realized - expected / total;
  }

  /** ツモで和了した席が得る点（包は考えない）。ほかの全ての席の払い ＋ 本場 ＋ 供託。 */
  static int tsumoGain(Variant variant, int base, int winner, int dealer, int honba, int sticks) {
    int paid = 0;
    for (int payer = 0; payer < variant.seats(); payer++)
      if (payer != winner) paid += ScoreMath.tsumo(base, winner == dealer, payer == dealer);
    return paid + Variant.HONBA * (variant.seats() - 1) * honba + Variant.DEPOSIT * sticks;
  }

  /** ツモのとき、払う席が払う点（本場を含む）。 */
  static int tsumoPayment(int base, int winner, int payer, int dealer, int honba) {
    return ScoreMath.tsumo(base, winner == dealer, payer == dealer) + Variant.HONBA * honba;
  }

  /** 栄和で放銃した席が払う点（本場を含む）。和了した席が得る点は、これに供託を足したもの。 */
  static int ronPayment(Variant variant, int base, int winner, int dealer, int honba) {
    return ScoreMath.ron(base, winner == dealer) + variant.ronHonba() * honba;
  }

  /**
   * 裏ドラの項に掛ける、席 {@code seat} から見た重みを返す（和了した席 {@code winner}、親 {@code dealer}）。ツモなら払いの割合から
   * 求める。
   */
  static double uraWeight(Variant variant, RoundOutcome outcome, int winner, int seat, int dealer) {
    if (seat == winner) return 1;
    if (outcome.kind() == EndKind.RON) return seat == outcome.loser() ? -1 : 0;
    double share = 0, all = 0;
    for (int payer = 0; payer < variant.seats(); payer++) {
      if (payer == winner) continue;
      int paid = ScoreMath.tsumo(SHARE_BASE, winner == dealer, payer == dealer);
      all += paid;
      if (payer == seat) share = paid;
    }
    return -share / all;
  }

  /**
   * 席の栄和の表。作った後に変えず、写しと分岐は親と共有する。
   *
   * @param stamp 表が依る席の状態のハッシュ
   * @param ron その席が栄和できる種類のビット集合（役の有無と振聴も考える）
   * @param ronBase 種類ごとの栄和の基本点（裏ドラは数えない。栄和できない種類は 0）
   */
  record SeatTable(long stamp, long ron, int[] ronBase) {}

  /**
   * 判断の時点で読んだイベント（テストの {@link #trace} 専用）。
   *
   * @param kind イベントの種類
   * @param seat ツモった席（表示は −1）
   * @param tile 引いた牌か表示牌の物理 ID
   * @param pool イベントの前のプール
   * @param hands イベントの前の、席ごとの純手牌の枚数
   */
  record Event(ChanceKind kind, int seat, int tile, int[] pool, int[][] hands) {}

  /** 1 スレッド・1 ルールの作業領域。 */
  private static final class Workspace {
    final LedgerTables tables;
    final ShapeAnalyzer shapes;

    /** ツモった席の、ツモの前の手牌。 */
    final HandShape before = new HandShape();

    /** 席の手牌を読み出す写し。 */
    final HandShape scratch = new HandShape();

    final WinScore score = new WinScore();
    final int[] pool = new int[Tiles.KINDS];

    /** 席ごとに持つ牌（純手牌・面子・抜いた北）の枚数。 */
    final int[][] owned;

    /** 栄和の表を作る席が持つ牌の枚数。 */
    final int[] held = new int[Tiles.KINDS];

    /** 席ごとの、この判断の時点の増分。 */
    final double[][] delta;

    /** 槓ドラの表示 1 回で席ごとに増えたドラの枚数と、プールの上での平均との差。 */
    final double[] deviation;

    /** 種類ごとの c の値（mask の上だけ書いて読む）。 */
    final double[] values = new double[Tiles.KINDS];

    /** 種類ごとのツモの和了の基本点（ツモで和了できる種類だけ書く）。 */
    final int[] tsumoBase = new int[Tiles.KINDS];

    /** このツモで使う、席ごとの栄和の表（経路の表を参照するだけで、中身は変えない）。 */
    final SeatTable[] seen;

    Workspace(LedgerTables tables) {
      this.tables = tables;
      shapes = new ShapeAnalyzer(tables.variant);
      owned = new int[tables.seats][Tiles.KINDS];
      delta = new double[tables.seats][COLUMNS];
      deviation = new double[tables.seats];
      seen = new SeatTable[tables.seats];
    }
  }
}
