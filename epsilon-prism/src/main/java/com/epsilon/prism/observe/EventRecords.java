package com.epsilon.prism.observe;

import static com.epsilon.prism.observe.ObservationSchema.DISCARD;
import static com.epsilon.prism.observe.ObservationSchema.EVENT;
import static com.epsilon.prism.observe.ObservationSchema.MELD;

import com.epsilon.core.GameState;
import com.epsilon.core.PublicEventLog;
import com.epsilon.core.RoundLimits;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.luck.LuckSchema;
import com.epsilon.prism.observe.ObservationSchema.DiscardBits;
import com.epsilon.prism.observe.ObservationSchema.EventBits;
import com.epsilon.prism.observe.ObservationSchema.EventKind;
import com.epsilon.prism.observe.ObservationSchema.MeldBits;

/**
 * エンジンの公開の出来事の記録（{@link PublicEventLog}）を O と O° の出来事の記録へ書く唯一のクラス（{@link ObservationWriter} と
 * {@code LuckWriter} が共有する）。歴史を河と手から組み立て直さず、記録を 1 回だけ前から読む。
 *
 * <p>書き先は 2 つで、欄が違う（だから 2 つの schema の enum をそれぞれ使う）:
 *
 * <ul>
 *   <li>O（{@link #writeObservation}）: 出来事の番号の順。席は観測者から見た相対席、鳴いた席は相対席か空（−1）。番号・巡目・河の位置は
 *       書かない（番号は記録の位置そのもの）。ポンの出来事は後に加槓してもポンの面子で、加槓は自分の出来事に加槓の面子を書く。
 *   <li>O°（{@link #writeLuck}）: 行為者から見た相対席の順に、席の中は河の順・副露の順。出来事の番号（絶対）を持ち、鳴かれたは 2 値。
 *       副露の記録は今の面子（加槓したポンは加槓の面子）と、作った出来事・加槓の出来事の番号。
 * </ul>
 *
 * <p>鳴かれた打牌と副露の整合（鳴きは元の打牌の直後、加槓は自分のポンから）はエンジンが記録を書くときに確かめるので、ここでは確かめない。
 * 書き先は 0 で始まり、記録は語に詰めてから丸ごと書く（{@link BitLayout#putRecord}）。割り当てはしない。
 * O° の席ごとの位置の作業域を持つので、1 スレッド専用（書き手ごとに 1 つ持つ）。
 */
public final class EventRecords {

  private static final int SEATS = GameState.NUM_PLAYERS;

  /** O の鳴いた席の空（鳴かれていない打牌）。 */
  private static final int NO_CALLER = -1;

  /** O° の加槓の出来事の番号の空（加槓していない副露）。 */
  private static final int NO_KAKAN = -1;

  /** O° の席ごとの次の副露の記録の番号。 */
  private final int[] meldCursor = new int[SEATS];

  /** O° の席ごとの次の打牌の記録の番号。 */
  private final int[] discardCursor = new int[SEATS];

  /** O° のポンの出来事の番号から、その副露の記録の番号（後の加槓がその記録を書き直す）。 */
  private final byte[] meldRecord = new byte[RoundLimits.MAX_PUBLIC_EVENTS];

  /**
   * O の出来事の節に、記録の全ての出来事を番号の順に書く。
   *
   * @param log 局の公開の出来事の記録（呼び出しの間だけ借りる）
   * @param observer 観測者の絶対席
   * @param words 書く先（0 で始まる）
   * @param at 出来事の節の先頭
   */
  static void writeObservation(PublicEventLog log, int observer, long[] words, int at) {
    for (int event = 0, events = log.size(); event < events; event++) {
      int author = relative(log.seat(event), observer);
      long record;
      if (log.isDiscard(event)) {
        int calledBy = log.calledBy(event);
        record = eventHead(EventKind.DISCARD, author);
        record = DISCARD.with(record, DiscardBits.TILE, log.tile(event));
        record = DISCARD.with(record, DiscardBits.AKA, log.aka(event) ? 1 : 0);
        record = DISCARD.with(record, DiscardBits.TSUMOGIRI, log.tsumogiri(event) ? 1 : 0);
        record = DISCARD.with(record, DiscardBits.RIICHI, log.riichi(event) ? 1 : 0);
        record =
            DISCARD.with(
                record,
                DiscardBits.CALLED_BY,
                calledBy == PublicEventLog.NOT_CALLED ? NO_CALLER : relative(calledBy, observer));
      } else {
        record = eventHead(EventKind.MELD, author);
        record = MELD.with(record, MeldBits.MELD, log.meld(event).canonicalId());
      }
      EVENT.putRecord(words, at, event, record);
    }
  }

  /**
   * O° の副露と河の節に、記録の全ての出来事を行為者から見た相対席の順に書く。
   *
   * <p>不変条件: {@code melds[s]} と {@code discards[s]} は記録の中の相対席 s の副露（加槓を除く）と打牌の出来事の数に等しい。
   * 局の歴史は記録だけが持ち、手の副露の数と河の長さは記録から作られるので、エンジンが保証する。崩れると記録が隣の席の位置へ
   * 黙って入るので、呼び出し側は数を同じ局面から取る。
   *
   * @param log 局の公開の出来事の記録（呼び出しの間だけ借りる）
   * @param actor 行為者の絶対席
   * @param melds 相対席ごとの副露の記録の数（席の節に書いた数）
   * @param discards 相対席ごとの打牌の記録の数（席の節に書いた数）
   * @param words 書く先（0 で始まる）
   * @param meldAt 副露の節の先頭
   * @param discardAt 河の節の先頭
   */
  public void writeLuck(
      PublicEventLog log,
      int actor,
      int[] melds,
      int[] discards,
      long[] words,
      int meldAt,
      int discardAt) {
    int meldStart = 0;
    int discardStart = 0;
    for (int seat = 0; seat < SEATS; seat++) {
      meldCursor[seat] = meldStart;
      discardCursor[seat] = discardStart;
      meldStart += melds[seat];
      discardStart += discards[seat];
    }
    for (int event = 0, events = log.size(); event < events; event++) {
      int seat = relative(log.seat(event), actor);
      if (log.isDiscard(event)) {
        int record = discardCursor[seat]++;
        LuckSchema.DISCARD.putRecord(words, discardAt, record, luckDiscard(log, event));
        continue;
      }
      int pon = log.ponEvent(event);
      if (pon == PublicEventLog.NO_PON) {
        int record = meldCursor[seat]++;
        meldRecord[event] = (byte) record;
        LuckSchema.MELD.putRecord(words, meldAt, record, luckMeld(log, event, event, NO_KAKAN));
      } else {
        LuckSchema.MELD.putRecord(words, meldAt, meldRecord[pon], luckMeld(log, event, pon, event));
      }
    }
  }

  /** O の出来事の記録の共通の頭（種類と作者）を詰めた語。 */
  private static long eventHead(EventKind kind, int author) {
    return EVENT.with(EVENT.with(0L, EventBits.KIND, kind.ordinal()), EventBits.AUTHOR, author);
  }

  /** O° の打牌の記録の語。 */
  private static long luckDiscard(PublicEventLog log, int event) {
    long record = LuckSchema.DISCARD.with(0L, LuckSchema.DiscardBits.TILE, log.tile(event));
    record = LuckSchema.DISCARD.with(record, LuckSchema.DiscardBits.AKA, log.aka(event) ? 1 : 0);
    record =
        LuckSchema.DISCARD.with(
            record, LuckSchema.DiscardBits.TSUMOGIRI, log.tsumogiri(event) ? 1 : 0);
    record =
        LuckSchema.DISCARD.with(record, LuckSchema.DiscardBits.RIICHI, log.riichi(event) ? 1 : 0);
    record =
        LuckSchema.DISCARD.with(
            record,
            LuckSchema.DiscardBits.CALLED,
            log.calledBy(event) == PublicEventLog.NOT_CALLED ? 0 : 1);
    return LuckSchema.DISCARD.with(record, LuckSchema.DiscardBits.EVENT, event);
  }

  /**
   * O° の副露の記録の語。
   *
   * @param meldEvent 面子を読む出来事（加槓は加槓の出来事）
   * @param created 面子を作った出来事（加槓は元のポンの出来事）
   * @param kakan 加槓の出来事（加槓でなければ {@link #NO_KAKAN}）
   */
  private static long luckMeld(PublicEventLog log, int meldEvent, int created, int kakan) {
    long record =
        LuckSchema.MELD.with(0L, LuckSchema.MeldBits.MELD, log.meld(meldEvent).canonicalId());
    record = LuckSchema.MELD.with(record, LuckSchema.MeldBits.EVENT, created);
    return LuckSchema.MELD.with(record, LuckSchema.MeldBits.KAKAN_EVENT, kakan);
  }

  /** 絶対席 {@code seat} の、{@code origin} から見た相対席（0 本人、1 下家、2 対面、3 上家）。 */
  private static int relative(int seat, int origin) {
    return Math.floorMod(seat - origin, SEATS);
  }
}
