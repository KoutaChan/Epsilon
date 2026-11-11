package com.epsilon.classic.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 判断の記録にある公開のイベントを 1 回だけ走査し、席ごとの河と面子、それぞれのイベントの番号を求める。席は判断する席から見た相対席。
 * 判断ごとに {@link #load} で作り直す。メモリは確保しない。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>イベントの番号は、判断の記録の中での番号（打牌・鳴き・暗槓・加槓が起きた順）。河には鳴かれた打牌も入る。トークンにするのは各席の
 * 河の先頭 {@link ClassicSchema#RIVER_TOKENS} 枚だけだが、河の枚数は全ての打牌で数える。面子は成立した順に並べ、ポンを加槓したら
 * ポンの位置のまま加槓の面子に置き換える。巡目は {@link RoundRules#afterDraw} の数え方で数え、チー・ポンの後の打牌は数えない。
 */
final class EventScan {

  /** 番号や河の位置が無いことを表す値。 */
  static final int NONE = -1;

  private static final int SEATS = ClassicSchema.SEATS;
  private static final int RIVER = ClassicSchema.RIVER_TOKENS;
  private static final int MELDS = ClassicSchema.MELD_TOKENS;

  private final Observation.EventRecord record = new Observation.EventRecord();
  private final int[] riverSize = new int[SEATS];

  /** 河にある牌の枚数。添字は 席 × 34 ＋ 種類。 */
  private final int[] riverCounts = new int[SEATS * Tiles.KINDS];

  /** 河のトークンになる打牌の、イベントの番号。添字は 席 × 24 ＋ 位置。下の配列も同じ並びで、その打牌の牌・巡目などを持つ。 */
  private final int[] riverEvent = new int[SEATS * RIVER];

  private final int[] riverTile = new int[SEATS * RIVER];
  private final int[] riverTurn = new int[SEATS * RIVER];
  private final int[] riverSequence = new int[SEATS * RIVER];
  private final boolean[] riverAka = new boolean[SEATS * RIVER];
  private final boolean[] riverTsumogiri = new boolean[SEATS * RIVER];
  private final boolean[] riverRiichi = new boolean[SEATS * RIVER];
  private final boolean[] riverCalled = new boolean[SEATS * RIVER];

  private final int[] riichiIndex = new int[SEATS];
  private final int[] riichiEvent = new int[SEATS];
  private final RiichiState[] declarations = new RiichiState[SEATS];
  private final int[] afterRiichi = new int[SEATS];
  private final int[] tsumogiriAfterRiichi = new int[SEATS];

  private final int[] meldCount = new int[SEATS];

  /**
   * 今の面子。添字は 席 × 4 ＋ 位置。下の配列も同じ並びで、成立したイベント、加槓したイベント、鳴いた時点と槓子になった時点の河の
   * 打牌の数を持つ。
   */
  private final Meld[] melds = new Meld[SEATS * MELDS];

  private final int[] meldEvent = new int[SEATS * MELDS];
  private final int[] kakanEvent = new int[SEATS * MELDS];
  private final int[] callRiver = new int[SEATS * MELDS];
  private final int[] kanRiver = new int[SEATS * MELDS];

  /** 読んだイベントのうち、打牌の数。下のフィールドは順に、ツモの後の打牌の数、打牌でないイベント（鳴き・暗槓・加槓）の数、槓子の数。 */
  private int discards;

  private int drawnDiscards;
  private int interruptions;
  private int kans;

  /** 判断の時点の河と面子を記録から作る。加槓の元のポンが無いとき、または面子が 4 つを超えるときは IllegalStateException を投げる。 */
  void load(Observation o) {
    Arrays.fill(riverSize, 0);
    Arrays.fill(riverCounts, 0);
    Arrays.fill(riichiIndex, NONE);
    Arrays.fill(riichiEvent, NONE);
    Arrays.fill(declarations, RiichiState.NONE);
    Arrays.fill(afterRiichi, 0);
    Arrays.fill(tsumogiriAfterRiichi, 0);
    Arrays.fill(meldCount, 0);
    discards = 0;
    drawnDiscards = 0;
    interruptions = 0;
    kans = 0;
    for (int event = 0; event < o.events(); event++) {
      o.readEvent(event, record);
      if (record.kind() == EventKind.DISCARD) {
        discard(event, RoundRules.afterDraw(o, event));
      } else {
        interruptions++;
        meld(event, record.author(), record.meld());
      }
    }
  }

  /** 巡目。ツモの後の打牌の数を席の数で割り、切り捨てた値。 */
  int turn() {
    return drawnDiscards / SEATS;
  }

  /** 全ての席の槓子の数。 */
  int kans() {
    return kans;
  }

  /** 河の打牌の数（鳴かれた打牌を含む）。 */
  int riverSize(int seat) {
    return riverSize[seat];
  }

  /** 河のその種類の枚数。 */
  int riverCount(int seat, int kind) {
    return riverCounts[seat * Tiles.KINDS + kind];
  }

  /** 河のトークンの数（河の先頭から 24 枚まで）。 */
  int riverTokens(int seat) {
    return Math.min(riverSize[seat], RIVER);
  }

  int riverEvent(int seat, int index) {
    return riverEvent[seat * RIVER + index];
  }

  int riverTile(int seat, int index) {
    return riverTile[seat * RIVER + index];
  }

  /** 打ったときの巡目。その打牌より前にあった、ツモの後の打牌の数を席の数で割り、切り捨てた値。 */
  int riverTurn(int seat, int index) {
    return riverTurn[seat * RIVER + index];
  }

  /** 局の中での打牌の通し番号（0 始まり）。 */
  int riverSequence(int seat, int index) {
    return riverSequence[seat * RIVER + index];
  }

  boolean riverAka(int seat, int index) {
    return riverAka[seat * RIVER + index];
  }

  boolean riverTsumogiri(int seat, int index) {
    return riverTsumogiri[seat * RIVER + index];
  }

  boolean riverRiichi(int seat, int index) {
    return riverRiichi[seat * RIVER + index];
  }

  boolean riverCalled(int seat, int index) {
    return riverCalled[seat * RIVER + index];
  }

  /** 立直の宣言牌の河の位置。宣言していなければ −1。 */
  int riichiIndex(int seat) {
    return riichiIndex[seat];
  }

  /** 立直の宣言牌のイベントの番号。宣言していなければ −1。 */
  int riichiEvent(int seat) {
    return riichiEvent[seat];
  }

  /** 宣言牌で決まる立直の状態（{@link RoundRules#declaration}）。宣言していなければ {@code NONE}。 */
  RiichiState declaration(int seat) {
    return declarations[seat];
  }

  /** 立直の宣言牌より後の、その席の打牌の数。 */
  int afterRiichi(int seat) {
    return afterRiichi[seat];
  }

  /** 立直の宣言牌より後の、その席のツモ切りの数。 */
  int tsumogiriAfterRiichi(int seat) {
    return tsumogiriAfterRiichi[seat];
  }

  int meldCount(int seat) {
    return meldCount[seat];
  }

  /** 今の面子（加槓したポンは加槓の面子）。 */
  Meld meld(int seat, int index) {
    return melds[seat * MELDS + index];
  }

  /** 面子が成立したイベントの番号（加槓したポンはポンのイベント）。 */
  int meldEvent(int seat, int index) {
    return meldEvent[seat * MELDS + index];
  }

  /** 加槓のイベントの番号。加槓でなければ −1。 */
  int kakanEvent(int seat, int index) {
    return kakanEvent[seat * MELDS + index];
  }

  /** チー・ポン（後で加槓したものを含む）を鳴いた時点で、面子の持ち主の河にあった打牌の数。ほかの面子は −1。 */
  int callRiver(int seat, int index) {
    return callRiver[seat * MELDS + index];
  }

  /** 槓子になった時点で、面子の持ち主の河にあった打牌の数（加槓は加槓した時点）。槓子でなければ −1。 */
  int kanRiver(int seat, int index) {
    return kanRiver[seat * MELDS + index];
  }

  /** 門前を崩す面子の数。 */
  int openMelds(int seat) {
    int open = 0;
    for (int index = 0; index < meldCount[seat]; index++) {
      if (!melds[seat * MELDS + index].type().concealed()) {
        open++;
      }
    }
    return open;
  }

  /** 槓子の数。 */
  int kans(int seat) {
    int count = 0;
    for (int index = 0; index < meldCount[seat]; index++) {
      if (melds[seat * MELDS + index].type().kan()) {
        count++;
      }
    }
    return count;
  }

  /**
   * 読んだ打牌のイベントを河に足す。
   *
   * @param event イベントの番号
   * @param drawn ツモの後の打牌か（そうなら巡目を進める）
   */
  private void discard(int event, boolean drawn) {
    int seat = record.author();
    int place = riverSize[seat]++;
    int kind = record.discardTile();
    riverCounts[seat * Tiles.KINDS + kind]++;
    if (place < RIVER) {
      int at = seat * RIVER + place;
      riverEvent[at] = event;
      riverTile[at] = kind;
      riverTurn[at] = drawnDiscards / SEATS;
      riverSequence[at] = discards;
      riverAka[at] = record.discardAka();
      riverTsumogiri[at] = record.discardTsumogiri();
      riverRiichi[at] = record.discardRiichi();
      riverCalled[at] = record.discardCalledBy() >= 0;
    }
    if (riichiIndex[seat] == NONE && record.discardRiichi()) {
      riichiIndex[seat] = place;
      riichiEvent[seat] = event;
      declarations[seat] = RoundRules.declaration(place == 0, interruptions > 0);
    } else if (riichiIndex[seat] != NONE) {
      afterRiichi[seat]++;
      if (record.discardTsumogiri()) {
        tsumogiriAfterRiichi[seat]++;
      }
    }
    discards++;
    if (drawn) {
      drawnDiscards++;
    }
  }

  private void meld(int event, int seat, Meld meld) {
    int first = seat * MELDS;
    if (meld.type() == Meld.Type.KAKAN) {
      for (int at = first; at < first + meldCount[seat]; at++) {
        if (melds[at].type() == Meld.Type.PON && melds[at].base() == meld.base()) {
          melds[at] = meld;
          kakanEvent[at] = event;
          kanRiver[at] = riverSize[seat];
          kans++;
          return;
        }
      }
      throw new IllegalStateException("Seat " + seat + " adds " + meld + " without its pon");
    }
    if (meldCount[seat] == MELDS) {
      throw new IllegalStateException("Seat " + seat + " holds more than " + MELDS + " melds");
    }
    int at = first + meldCount[seat]++;
    melds[at] = meld;
    meldEvent[at] = event;
    kakanEvent[at] = NONE;
    boolean call = meld.type() == Meld.Type.CHI || meld.type() == Meld.Type.PON;
    callRiver[at] = call ? riverSize[seat] : NONE;
    kanRiver[at] = meld.type().kan() ? riverSize[seat] : NONE;
    if (meld.type().kan()) {
      kans++;
    }
  }
}
