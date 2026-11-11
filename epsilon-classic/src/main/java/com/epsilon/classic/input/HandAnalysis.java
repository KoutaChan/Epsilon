package com.epsilon.classic.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 自分の手（{@link OwnHand} と立直の状態）の形と待ちを求める。判断の時点の手にも、合法手の後の手にも使う。手を読み込むか合法手の後の
 * 手にしてから、{@link #analyzeShape} か {@link #analyze} で求める。メモリは確保しない。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>受け入れと待ちを持つのは、13 枚相当の手だけ。受け入れに入るのはまだ引ける種類だけで、純手牌と面子でその種類の枚数（ルールで
 * 決まる）を使い切った種類は除く。受け入れの枚数は判断の時点で自分から見えない枚数で数えるので、どの合法手の後でも変わらない（打牌と
 * 鳴きは、牌を手から公開の場所へ移すだけだから）。
 *
 * <p>待ちの点は、和了牌を赤でない牌として数え、立直は手の立直の状態のまま数える。一発・嶺上・海底・河底・槍槓・天和・地和と裏ドラは
 * 数えない。立直していれば、立直を数えなくても役があるかも求める。次のツモで役のある聴牌にできるかは、1 向聴の 13 枚相当の手だけ
 * について、立直を数えずに採点して求める。
 */
final class HandAnalysis {

  /** 役があるかを探す和了の仕方を表すビット（ロンとツモ）。 */
  private static final int RON = 1;

  private static final int TSUMO = 2;

  private final Variant variant;
  private final TileSet set;
  private final ShapeAnalyzer analyzer;
  private final HandScorer scorer = new HandScorer();

  /** 待ちの点を求める条件。 */
  private final WinContext scoring;

  /** 次のツモの後に役があるかを探す条件。 */
  private final WinContext probing;

  private final WinScore scratch = new WinScore();
  private final HandShape probe = new HandShape();

  private final OwnHand hand;
  private RiichiState riichi = RiichiState.NONE;

  /** 種類ごとの、面子にある枚数。 */
  private final int[] meldTiles = new int[Tiles.KINDS];

  private long meldKinds;

  private int shanten;
  private int standardShanten;
  private int chiitoitsuShanten;
  private int kokushiShanten;
  private boolean special;
  private long improving;
  private long waits;
  private int waitCount;
  private final int[] waitKinds = new int[ClassicSchema.MAX_WAITS];
  private final WinScore[] ron = new WinScore[ClassicSchema.MAX_WAITS];
  private final WinScore[] tsumo = new WinScore[ClassicSchema.MAX_WAITS];
  private long ronWaits;
  private long tsumoWaits;
  private long intrinsicRon;
  private long intrinsicTsumo;
  private long freeRon;
  private long furitenRon;
  private long tsumoReach;

  HandAnalysis(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    analyzer = new ShapeAnalyzer(variant);
    scoring = new WinContext(variant);
    probing = new WinContext(variant);
    hand = new OwnHand(variant);
    for (int wait = 0; wait < ClassicSchema.MAX_WAITS; wait++) {
      ron[wait] = new WinScore();
      tsumo[wait] = new WinScore();
    }
  }

  /**
   * 判断の時点の自分の手を読み込む。
   *
   * @param o 判断の記録（立直の状態を読む）
   * @param tiles 同じ記録から作った牌の枚数（純手牌と赤を読む）
   * @param facts 同じ記録から作った席の事実（面子を読む）
   */
  void load(Observation o, TileCounts tiles, SeatFacts facts) {
    hand.load(tiles, facts);
    riichi = o.riichi(0);
  }

  /**
   * 合法手の後の手にする。解析の結果は、求め直すまで前の手のものが残る。
   *
   * @param current 判断の時点の手
   * @param legal 判断の時点の合法手
   * @param meld 合法手が作る面子（判断の時点の手で {@link OwnHand#meldOf} が返す値。作らない手は null）
   * @param discard チー・ポンの後に打つ牌（ほかの手は読まない）
   * @param riichi 合法手の後の立直の状態
   */
  void play(HandAnalysis current, Action legal, Meld meld, Action discard, RiichiState riichi) {
    hand.play(current.hand, legal, meld, discard);
    this.riichi = Objects.requireNonNull(riichi, "riichi");
  }

  /** 純手牌・面子・純手牌の赤。次に書き換えるまで有効。 */
  OwnHand hand() {
    return hand;
  }

  /** 向聴・受け入れ・待ちを求める。点は求めない。 */
  void analyzeShape() {
    meldKinds = 0;
    Arrays.fill(meldTiles, 0);
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      for (int tile = 0; tile < meld.size(); tile++) {
        meldTiles[meld.tile(tile)]++;
      }
      meldKinds |= meld.kinds();
    }
    HandShape shape = hand.shape();
    shanten = analyzer.shanten(shape);
    standardShanten = analyzer.standardShanten(shape);
    special = shape.melds() == 0;
    chiitoitsuShanten = special ? ShapeAnalyzer.chiitoitsuShanten(shape) : 0;
    kokushiShanten = special ? ShapeAnalyzer.kokushiShanten(shape) : 0;
    improving = drawing() ? analyzer.improving(shape) & ~exhausted(shape) : 0;
    waits = drawing() && shanten == 0 ? improving : 0;
  }

  /**
   * 手の形、待ちの点、次のツモで役のある聴牌にできる種類を求める。
   *
   * @param o 判断の記録（風とドラ表示牌を読む）
   * @param tiles 同じ記録から作った牌の枚数
   * @param river 遷移の後の自分の河にある種類（遷移で打つ種類を含む）
   */
  void analyze(Observation o, TileCounts tiles, long river) {
    analyzeShape();
    int round = o.kyoku();
    int seatWind = variant.seatWind(round, o.observer());
    int roundWind = variant.roundWind(round);
    scoreWaits(o, seatWind, roundWind);
    frontier(tiles, river, seatWind, roundWind);
  }

  int shanten() {
    return shanten;
  }

  int standardShanten() {
    return standardShanten;
  }

  /** 面子が無く、七対子と国士の形も数える手か。 */
  boolean special() {
    return special;
  }

  /** 七対子の向聴（{@link #special} でなければ 0）。 */
  int chiitoitsuShanten() {
    return chiitoitsuShanten;
  }

  /** 国士の向聴（{@link #special} でなければ 0）。 */
  int kokushiShanten() {
    return kokushiShanten;
  }

  /** 引くと向聴が下がる種類のうち、まだ引ける種類。 */
  long improving() {
    return improving;
  }

  /** 形の上の待ち（聴牌の手の受け入れ）。 */
  long waits() {
    return waits;
  }

  int waitCount() {
    return waitCount;
  }

  /**
   * 待ちの種類。
   *
   * @param index 待ちの番号（種類の昇順）
   */
  int waitKind(int index) {
    return waitKinds[index];
  }

  /**
   * 待ちのロンの点。役が無ければ基本点 0。次の解析まで有効。
   *
   * @param index 待ちの番号（種類の昇順）
   */
  WinScore ron(int index) {
    return ron[index];
  }

  /**
   * 待ちのツモの点。役が無ければ基本点 0。次の解析まで有効。
   *
   * @param index 待ちの番号（種類の昇順）
   */
  WinScore tsumo(int index) {
    return tsumo[index];
  }

  /** ロンで役のある待ち。 */
  long ronWaits() {
    return ronWaits;
  }

  /** ツモで役のある待ち。 */
  long tsumoWaits() {
    return tsumoWaits;
  }

  /** 立直を数えなくてもロンで役のある待ち。 */
  long intrinsicRon() {
    return intrinsicRon;
  }

  /** 立直を数えなくてもツモで役のある待ち。 */
  long intrinsicTsumo() {
    return intrinsicTsumo;
  }

  /** 引いて 1 枚打つと、振聴でなく、ロンで役のある聴牌にできる種類。 */
  long freeRon() {
    return freeRon;
  }

  /** 引いて 1 枚打つと、ロンで役のある聴牌にはなるが、どう打っても振聴になる種類。 */
  long furitenRon() {
    return furitenRon;
  }

  /** 引いて 1 枚打つと、ツモで役のある聴牌にできる種類。 */
  long tsumoReach() {
    return tsumoReach;
  }

  /** 包が成り立つ和了か。大三元では 3 つ目の三元牌の面子が、大四喜では 4 つ目の風牌の面子が（成立した順に数えて）鳴いた面子なら成り立つ。 */
  boolean pao(WinScore score) {
    if (score.has(ScoringYaku.DAISANGEN)) {
      return liable(Tiles.range(Tiles.HAKU, Tiles.CHUN));
    }
    if (score.has(ScoringYaku.DAISUUSHII)) {
      return liable(Tiles.range(Tiles.TON, Tiles.PEI));
    }
    return false;
  }

  /** 13 枚相当の手か。 */
  private boolean drawing() {
    return hand.shape().size() == Variant.HAND;
  }

  /**
   * 面子にある種類のうち、純手牌と面子でその種類の枚数（ルールで決まる）を使い切った種類。純手牌だけで使い切った種類は、形の解析の
   * 側で除く。
   */
  private long exhausted(HandShape shape) {
    long exhausted = 0;
    for (long kinds = meldKinds; kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      if (shape.count(kind) + meldTiles[kind] >= set.copies(kind)) {
        exhausted |= Tiles.bit(kind);
      }
    }
    return exhausted;
  }

  /** 待ちごとにロン・ツモの点を求める。 */
  private void scoreWaits(Observation o, int seatWind, int roundWind) {
    waitCount = 0;
    ronWaits = 0;
    tsumoWaits = 0;
    intrinsicRon = 0;
    intrinsicTsumo = 0;
    if (waits == 0) {
      return;
    }
    prepare(scoring, hand.shape(), seatWind, roundWind);
    for (int index = 0; index < o.indicators(); index++) {
      scoring.addIndicator(o.indicatorTile(index));
    }
    scoring.setRiichi(riichi, false);
    scoring.setRedFives(hand.reds());
    for (long rest = waits; rest != 0; rest &= rest - 1) {
      int kind = Long.numberOfTrailingZeros(rest);
      long bit = Tiles.bit(kind);
      int wait = waitCount++;
      waitKinds[wait] = kind;
      if (score(scoring, kind, Source.RON, ron[wait])) {
        ronWaits |= bit;
      }
      if (score(scoring, kind, Source.TSUMO, tsumo[wait])) {
        tsumoWaits |= bit;
      }
      if (riichi == RiichiState.NONE) {
        continue;
      }
      scoring.setRiichi(RiichiState.NONE, false);
      if (score(scoring, kind, Source.RON, scratch)) {
        intrinsicRon |= bit;
      }
      if (score(scoring, kind, Source.TSUMO, scratch)) {
        intrinsicTsumo |= bit;
      }
      scoring.setRiichi(riichi, false);
    }
    if (riichi == RiichiState.NONE) {
      intrinsicRon = ronWaits;
      intrinsicTsumo = tsumoWaits;
    }
  }

  /** 1 向聴の 13 枚相当の手について、受け入れの種類ごとに、引いて 1 枚打つと役のある聴牌にできるかを求める。見えない牌の無い種類は飛ばす。 */
  private void frontier(TileCounts tiles, long river, int seatWind, int roundWind) {
    freeRon = 0;
    furitenRon = 0;
    tsumoReach = 0;
    if (!drawing() || shanten != 1) {
      return;
    }
    HandShape shape = hand.shape();
    prepare(probing, shape, seatWind, roundWind);
    long handKinds = shape.kinds();
    for (long draws = improving; draws != 0; draws &= draws - 1) {
      int draw = Long.numberOfTrailingZeros(draws);
      if (tiles.unseen(draw) == 0) {
        continue;
      }
      boolean free = false;
      boolean furiten = false;
      boolean reach = false;
      probe.copyFrom(shape);
      probe.add(draw);
      for (long discards = handKinds & ~Tiles.bit(draw); discards != 0 && !(free && reach); discards &= discards - 1) {
        int discard = Long.numberOfTrailingZeros(discards);
        probe.remove(discard);
        if (analyzer.shanten(probe) == 0) {
          long tenpai = analyzer.waits(probe) & ~exhausted(probe);
          boolean blocked = (tenpai & (river | Tiles.bit(discard))) != 0;
          boolean needRon = blocked ? !free && !furiten : !free;
          int found = tenpai == 0 ? 0 : probeYaku(tenpai, needRon, !reach);
          if ((found & RON) != 0) {
            if (blocked) {
              furiten = true;
            } else {
              free = true;
            }
          }
          reach |= (found & TSUMO) != 0;
        }
        probe.add(discard);
      }
      long bit = Tiles.bit(draw);
      if (free) {
        freeRon |= bit;
      } else if (furiten) {
        furitenRon |= bit;
      }
      if (reach) {
        tsumoReach |= bit;
      }
    }
  }

  /**
   * 試しに作った聴牌の手について、待ちのどれかでロン・ツモの役が付くかを探す。
   *
   * @param tenpai 試しの手の待ち
   * @param needRon ロンの役を探すか
   * @param needTsumo ツモの役を探すか
   * @return 役のあった和了の仕方のビット（{@code RON}・{@code TSUMO}）
   */
  private int probeYaku(long tenpai, boolean needRon, boolean needTsumo) {
    probing.hand().copyFrom(probe);
    int found = 0;
    boolean ronLeft = needRon;
    boolean tsumoLeft = needTsumo;
    for (long rest = tenpai; rest != 0 && (ronLeft || tsumoLeft); rest &= rest - 1) {
      int kind = Long.numberOfTrailingZeros(rest);
      if (ronLeft && score(probing, kind, Source.RON, scratch)) {
        found |= RON;
        ronLeft = false;
      }
      if (tsumoLeft && score(probing, kind, Source.TSUMO, scratch)) {
        found |= TSUMO;
        tsumoLeft = false;
      }
    }
    return found;
  }

  /** 採点の条件を空にして、純手牌・面子・風だけを書く。ドラ表示牌・立直・赤は書かない。 */
  private void prepare(WinContext context, HandShape shape, int seatWind, int roundWind) {
    context.clear();
    context.hand().copyFrom(shape);
    for (int index = 0; index < hand.meldCount(); index++) {
      context.addMeld(hand.meld(index));
    }
    context.setWinds(seatWind, roundWind);
  }

  private boolean score(WinContext context, int kind, Source source, WinScore out) {
    context.setWin(kind, source);
    return scorer.score(context, out);
  }

  /** 三元牌か風牌の面子を成立した順に数え、その種類の数（三元牌なら 3、風牌なら 4）に届いた面子が鳴いた面子かを返す。 */
  private boolean liable(long group) {
    int sets = 0;
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      if (Tiles.contains(group, meld.base()) && ++sets == Long.bitCount(group)) {
        return meld.source() != Meld.RelativeSource.NONE;
      }
    }
    return false;
  }
}
