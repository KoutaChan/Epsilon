package com.epsilon.agent.observe;

import com.epsilon.agent.observe.ObservationSchema.CallDiscardBits;
import com.epsilon.agent.observe.ObservationSchema.DiscardBits;
import com.epsilon.agent.observe.ObservationSchema.EventBits;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.ObservationSchema.HandBits;
import com.epsilon.agent.observe.ObservationSchema.HeaderBits;
import com.epsilon.agent.observe.ObservationSchema.LastEvent;
import com.epsilon.agent.observe.ObservationSchema.LegalBits;
import com.epsilon.agent.observe.ObservationSchema.MeldBits;
import com.epsilon.agent.observe.ObservationSchema.SeatBits;
import com.epsilon.agent.observe.ObservationSchema.Section;
import com.epsilon.agent.observe.ObservationSchema.WinBits;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 記録 O（{@link ObservationSchema}）を読むリーダー。{@link #bind} で別の O に付け替えて使い回し、読むときにメモリを確保しない。
 * 付け替えて使うので、読むスレッドごとに 1 つ持つ。ルール（四人打ちか三人打ちか）は作るときに決まり、そのルールの O だけを読む。
 *
 * <p>席は観測席から見た相対席（0 が自分、1 が下家、…、S − 1 が上家）で返す。値が無いときは、牌・席・番号なら −1 を、enum なら null を
 * 返す。手と面子は、Action・Meld が共有の表に持つオブジェクトを返す。
 *
 * <p>O に書いていない値でも、使う側どうしで答えをそろえる必要がある値はここで求める。場風は局から、第一巡かどうか・副露の有無・第一打
 * の前かどうか・抜いた北の数はイベントから、ツモ牌は直前のイベントから求める。同値な打牌の代表、遷移の数、槓ドラの表示を置く位置の
 * 決め方もここに置く。O を書く側は、ヘッダーの遷移の数を {@link #countTransitions} で求めて書く。
 *
 * <p>イベントや合法手などの番号は範囲を確かめ、範囲の外なら IndexOutOfBoundsException を投げる。副露のイベントで打牌のフィールドを
 * 読むなど、種類の違うイベントのフィールドを読むと IllegalStateException を投げる。イベントのフィールドを全て読むときは、フィールド
 * ごとのメソッドではなく {@link #readEvent} で記録を 1 回だけ読む。
 */
public final class Observation {

  /** 同値な打牌の組の数（打牌か立直の打牌か × 牌の種類 × 赤かどうか）。 */
  private static final int TWIN_KEYS = 2 * Tiles.KINDS * 2;

  private final ObservationSchema schema;
  private final Variant variant;
  private final int seats;
  private final int discardKind;
  private final int meldKind;

  /** イベントの種類の並びで、北抜きが何番目か。北抜きの無いルールでは −1。 */
  private final int nukiKind;

  /** 相対席ごとの抜いた北の数。 */
  private final int[] nuki;
  private final int[] start = new int[ObservationSchema.sections()];
  private final Action[] legal = new Action[ObservationSchema.MAX_LEGAL];
  private final int[] representatives = new int[ObservationSchema.MAX_LEGAL];
  private boolean representativesFound;

  /**
   * 同値な打牌の組ごとに、最初に出た手。{@code twinSeen} には、組ごとにこの値を書いた走査の番号を持つ（番号が今の走査と違う組は、
   * まだ出ていない）。
   */
  private final int[] twinFirst = new int[TWIN_KEYS];

  private final int[] twinSeen = new int[TWIN_KEYS];
  private int twinStamp;
  private long[] words;
  private int base;
  private int longs;
  private int events;
  private int actions;
  private int callDiscards;
  private int wins;
  private boolean scanned;
  private int discardedSeats;
  private boolean meldMade;

  public Observation(Variant variant) {
    schema = ObservationSchema.of(variant);
    this.variant = variant;
    seats = variant.seats();
    discardKind = schema.eventKinds.index(EventKind.DISCARD);
    meldKind = schema.eventKinds.index(EventKind.MELD);
    nukiKind = schema.eventKinds.contains(EventKind.NUKI) ? schema.eventKinds.index(EventKind.NUKI) : -1;
    nuki = new int[seats];
  }

  public Variant variant() {
    return variant;
  }

  /** 配列の指定した位置から始まる O を読むように付け替える。O が配列に収まらなければ IndexOutOfBoundsException を投げる。 */
  public Observation bind(long[] words, int base) {
    this.words = Objects.requireNonNull(words, "words");
    this.base = base;
    longs = schema.place(words, base, start);
    Objects.checkFromIndexSize(base, longs, words.length);
    events = header(HeaderBits.EVENTS);
    actions = header(HeaderBits.LEGAL_ACTIONS);
    callDiscards = header(HeaderBits.CALL_DISCARDS);
    wins = header(HeaderBits.WINS);
    int legalStart = start(Section.LEGAL);
    for (int action = 0; action < actions; action++) {
      legal[action] = Action.of(schema.legal.get(words, legalStart, action, LegalBits.ACTION));
    }
    scanned = false;
    representativesFound = false;
    return this;
  }

  /** バッチの中の、指定した番号の O を読むように付け替える。バッチのルールが違えば IllegalArgumentException を投げる。 */
  public Observation bind(ObservationBatch batch, int index) {
    if (batch.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + batch.variant());
    }
    return bind(batch.words(), batch.start(index));
  }

  /** O が占める {@code long} の数。 */
  public int longs() {
    return longs;
  }

  public EngineDecisionKind decisionKind() {
    return schema.decisionKinds.get(header(HeaderBits.DECISION_KIND));
  }

  /** 観測席（絶対席）。 */
  public int observer() {
    return header(HeaderBits.OBSERVER);
  }

  /** 相対席を絶対席に直す。同点の順位を席の順で決めるときなどに使う。 */
  public int absoluteSeat(int seat) {
    return (observer() + Objects.checkIndex(seat, seats)) % seats;
  }

  /** 親の相対席。 */
  public int dealer() {
    return header(HeaderBits.DEALER);
  }

  /** 局の通し番号（東 1 局が 0。西場の最後の局まで数える）。 */
  public int kyoku() {
    return header(HeaderBits.KYOKU);
  }

  /** 場風の牌の種類。 */
  public int roundWindTile() {
    return variant.roundWind(kyoku());
  }

  public int honba() {
    return header(HeaderBits.HONBA);
  }

  /** 供託の本数。 */
  public int kyotaku() {
    return header(HeaderBits.KYOTAKU);
  }

  /** 王牌を除いた山の残りの枚数。 */
  public int remainingWall() {
    return header(HeaderBits.WALL);
  }

  /** 第一巡の途中なら true（まだ打牌していない席がある）。 */
  public boolean firstCycle() {
    scanEvents();
    return discardedSeats != variant.seatMask();
  }

  /** この局で副露（暗槓を含む）があれば true。北抜きは副露に数えない。 */
  public boolean callMade() {
    scanEvents();
    return meldMade;
  }

  /** 指定した相対席が、この局で抜いた北の数。北抜きの無いルールでは 0。 */
  public int nuki(int seat) {
    Objects.checkIndex(seat, seats);
    scanEvents();
    return nuki[seat];
  }

  /** 自分が見逃しで振聴になっていれば true（同巡内の振聴と立直の後の振聴を含み、河による振聴は含まない）。 */
  public boolean furitenTemporary() {
    return header(HeaderBits.FURITEN_TEMPORARY) != 0;
  }

  /** 直前のイベント（無ければ null）。 */
  public LastEvent lastEvent() {
    int index = header(HeaderBits.LAST_EVENT);
    return index < 0 ? null : schema.lastEvents.get(index);
  }

  /** 直前のイベントを起こした席（相対席。無ければ −1）。 */
  public int lastEventSeat() {
    return header(HeaderBits.LAST_EVENT_SEAT);
  }

  /** 直前のイベントの牌（イベントが無いときと、他家のツモのときは −1）。 */
  public int lastEventTile() {
    return header(HeaderBits.LAST_EVENT_TILE);
  }

  public boolean lastEventAka() {
    return header(HeaderBits.LAST_EVENT_AKA) != 0;
  }

  /** 直前のツモが嶺上牌なら true。 */
  public boolean lastEventRinshan() {
    return header(HeaderBits.LAST_EVENT_RINSHAN) != 0;
  }

  /** 直前に宣言した槓の種類（直前のイベントが槓の宣言でなければ null）。 */
  public TurnEvent.KanKind lastKanKind() {
    int index = header(HeaderBits.LAST_KAN_KIND);
    return index < 0 ? null : schema.kanKinds.get(index);
  }

  /** 大明槓・加槓の後めくりの槓ドラを、まだめくっていなければ true。 */
  public boolean deferredDora() {
    return header(HeaderBits.DEFERRED_DORA) != 0;
  }

  /** 今ツモった牌（直前のイベントが自分のツモでなければ −1）。値は直前のイベントの牌と同じ。 */
  public int drawnTile() {
    return ownDraw() ? lastEventTile() : -1;
  }

  /** 自分の純手牌にある赤を、色ごとのビットで表した値（ビット 0 が萬子、1 が筒子、2 が索子）。 */
  public int handAkaMask() {
    return header(HeaderBits.HAND_AKA);
  }

  /** ドラ表示牌の数（最初の表示牌を含むので 1 以上）。 */
  public int indicators() {
    return header(HeaderBits.INDICATORS);
  }

  /** 槓ドラの表示牌の数（最初の表示牌を除いたドラ表示牌の数）。 */
  public int kanDoraIndicators() {
    return indicators() - 1;
  }

  /** ドラ表示牌の種類。番号はめくった順で、0 が最初の表示牌。 */
  public int indicatorTile(int index) {
    return schema.header.get(words, base, 0, HeaderBits.INDICATOR, Objects.checkIndex(index, indicators()));
  }

  /** ドラ表示牌に出た赤を、色ごとのビットで表した値。どの表示牌が赤かは O からは分からない。 */
  public int indicatorAkaMask() {
    return header(HeaderBits.INDICATOR_AKA);
  }

  /** O に書いたイベントの数（打牌・副露・加槓・北抜き。槓ドラの表示は含まない）。 */
  public int events() {
    return events;
  }

  /** 公開のイベントの数（O のイベント ＋ 槓ドラの表示）。局の始めにめくる最初の表示牌は、イベントに数えない。 */
  public int publicEvents() {
    return events + kanDoraIndicators();
  }

  public int legalActions() {
    return actions;
  }

  /** 遷移の数（O を書く側が {@link #countTransitions} で求めて書いた値）。 */
  public int transitions() {
    return header(HeaderBits.TRANSITIONS);
  }

  /** この判断で選べる和了（ロン・ツモの合法手）の数。 */
  public int wins() {
    return wins;
  }

  /** 指定した相対席の点。 */
  public int score(int seat) {
    return ObservationSchema.decodePoints(seat(seat, SeatBits.SCORE));
  }

  public RiichiState riichi(int seat) {
    return schema.riichiStates.get(seat(seat, SeatBits.RIICHI));
  }

  public boolean ippatsu(int seat) {
    return seat(seat, SeatBits.IPPATSU) != 0;
  }

  /** 指定した相対席が、まだ第一打を打っていなければ true（その席の打牌のイベントが無い）。 */
  public boolean beforeFirstDiscard(int seat) {
    Objects.checkIndex(seat, seats);
    scanEvents();
    return (discardedSeats & 1 << seat) == 0;
  }

  /** 自分の純手牌にある、指定した種類の枚数。 */
  public int handCount(int kind) {
    return schema.hand.get(words, start(Section.HAND), 0, HandBits.COUNT, kind);
  }

  /** 自分の純手牌にある枚数を、渡した配列の先頭から牌の種類の順に書く。書くのは全ての種類の分（{@code Tiles.KINDS} 個）。 */
  public void handCounts(int[] counts) {
    schema.hand.getAll(words, start(Section.HAND), 0, HandBits.COUNT, counts);
  }

  public EventKind eventKind(int event) {
    return schema.eventKinds.get(schema.event.get(eventWord(event), EventBits.KIND));
  }

  /** イベントを起こした席（相対席）。 */
  public int eventAuthor(int event) {
    return schema.event.get(eventWord(event), EventBits.AUTHOR);
  }

  /**
   * イベントの記録を 1 回だけ読み、種類、起こした席、種類ごとのフィールドを書く先に入れる。
   *
   * @param event イベントの番号
   * @param out 書く先（呼び出し側が持って使い回す）
   * @return 渡された書く先
   */
  public EventRecord readEvent(int event, EventRecord out) {
    long word = eventWord(event);
    out.event = event;
    out.kind = schema.eventKinds.get(schema.event.get(word, EventBits.KIND));
    out.author = schema.event.get(word, EventBits.AUTHOR);
    if (out.kind == EventKind.DISCARD) {
      out.tile = schema.discard.get(word, DiscardBits.TILE);
      out.aka = schema.discard.get(word, DiscardBits.AKA) != 0;
      out.tsumogiri = schema.discard.get(word, DiscardBits.TSUMOGIRI) != 0;
      out.riichi = schema.discard.get(word, DiscardBits.RIICHI) != 0;
      out.calledBy = schema.discard.get(word, DiscardBits.CALLED_BY);
      out.meld = null;
    } else if (out.kind == EventKind.MELD) {
      out.meld = Meld.of(schema.meld.get(word, MeldBits.MELD));
    } else {
      out.meld = null;
    }
    return out;
  }

  /** 打牌のイベントで打った牌の種類。 */
  public int discardTile(int event) {
    return discard(event, DiscardBits.TILE);
  }

  public boolean discardAka(int event) {
    return discard(event, DiscardBits.AKA) != 0;
  }

  public boolean discardTsumogiri(int event) {
    return discard(event, DiscardBits.TSUMOGIRI) != 0;
  }

  /** 打牌のイベントで打った牌が立直の宣言牌なら true。 */
  public boolean discardRiichi(int event) {
    return discard(event, DiscardBits.RIICHI) != 0;
  }

  /** 打牌を鳴いた席（相対席。鳴かれていなければ −1）。鳴いた副露は、次の番号のイベントになる。 */
  public int discardCalledBy(int event) {
    return discard(event, DiscardBits.CALLED_BY);
  }

  /** 副露・加槓のイベントの時点での面子。後で加槓したポンも、ポンのイベントではポンのまま返す。 */
  public Meld meld(int event) {
    long word = eventWord(event);
    requireKind(event, word, EventKind.MELD);
    return Meld.of(schema.meld.get(word, MeldBits.MELD));
  }

  /**
   * 槓ドラの表示を置く位置を返す。エンジンは、暗槓なら槓のイベントの直後に、大明槓・加槓なら槓をした席の次のイベント（打牌か次の
   * 槓）の直後に槓ドラをめくる。その位置のイベントが無ければ IllegalStateException を投げる。
   *
   * @param kan 何番目の槓か（イベントの順。0 から数える）。その槓ドラの表示牌は {@code indicatorTile(kan + 1)} で読める
   * @return 直後に槓ドラの表示を置くイベントの番号
   */
  public int kanDoraEvent(int kan) {
    Objects.checkIndex(kan, kanDoraIndicators());
    int seen = 0;
    for (int event = 0; event < events; event++) {
      if (eventKind(event) != EventKind.MELD) {
        continue;
      }
      Meld meld = meld(event);
      if (!meld.type().kan() || seen++ < kan) {
        continue;
      }
      if (meld.type() == Meld.Type.ANKAN) {
        return event;
      }
      int author = eventAuthor(event);
      for (int next = event + 1; next < events; next++) {
        if (eventAuthor(next) == author) {
          return next;
        }
      }
      throw new IllegalStateException("Kan dora " + kan + " is shown before its author acts again");
    }
    throw new IllegalStateException("Kan dora " + kan + " is shown without its kan");
  }

  public Action action(int action) {
    return legal[Objects.checkIndex(action, actions)];
  }

  /** 指定した合法手の後に打てる打牌の数（チー・ポンだけ。ほかの手は 0）。 */
  public int callDiscardsOf(int action) {
    return schema.legal.get(words, start(Section.LEGAL), Objects.checkIndex(action, actions), LegalBits.CALL_DISCARDS);
  }

  /** チー・ポンの後の打牌を返す。番号は、チー・ポンの合法手の順に後の打牌を並べ、通しで振ったもの。 */
  public Action callDiscard(int index) {
    return Action.of(
        schema.callDiscard.get(
            words, start(Section.CALL_DISCARD), Objects.checkIndex(index, callDiscards), CallDiscardBits.ACTION));
  }

  /**
   * 指定した合法手の代表を、合法手の番号で返す。同値な打牌とは、種類（打牌か立直の打牌か）・牌・赤の選び方が同じで、ツモ切り
   * かどうかだけが違う打牌のこと（{@link #twins}）。同値な打牌の中では番号の最も小さい手が代表になり、打牌でない手と、同値な
   * 打牌の無い打牌は、自分自身が代表になる。代表は、付け替えた後に初めて呼ばれたときに、全ての手の分をまとめて決める。
   */
  public int representative(int action) {
    Objects.checkIndex(action, actions);
    if (!representativesFound) {
      findRepresentatives();
    }
    return representatives[action];
  }

  /**
   * 2 つの手が同値な打牌かを返す。種類（打牌か立直の打牌か）・牌・赤の選び方が同じ打牌なら、ツモ切りかどうかに依らず true（同じ
   * 手どうしでも true）。打牌でない手は、同じ手どうしでも false。
   */
  public static boolean twins(Action action, Action other) {
    Action.Type type = action.type();
    return (type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI)
        && other.type() == type
        && other.tile() == action.tile()
        && other.selection().red() == action.selection().red();
  }

  /** 指定した合法手の遷移の数。代表でない同値な打牌は 0（代表の遷移を使う）、チー・ポンは後の打牌ごとに 1 つ、ほかは 1。 */
  public int transitionsOf(int action) {
    if (representative(action) != action) {
      return 0;
    }
    return ObservationSchema.hasCallDiscards(action(action).type()) ? callDiscardsOf(action) : 1;
  }

  /** 遷移の総数（全ての合法手について {@link #transitionsOf} を足したもの）。O を書く側が、ヘッダーの遷移の数に書く値。 */
  public int countTransitions() {
    int total = 0;
    for (int action = 0; action < actions; action++) {
      total += transitionsOf(action);
    }
    return total;
  }

  /** 合法手に対応する和了の番号（ロン・ツモでなければ −1）。和了の番号は、ロン・ツモの合法手の順に振る。 */
  public int winOf(int action) {
    if (!ObservationSchema.hasWin(action(action).type())) {
      return -1;
    }
    int win = 0;
    for (int other = 0; other < action; other++) {
      if (ObservationSchema.hasWin(legal[other].type())) {
        win++;
      }
    }
    return win;
  }

  /** 和了の翻（裏ドラを数えない。役満は 0）。 */
  public int winHan(int win) {
    return win(win, WinBits.HAN);
  }

  /** 和了の符のコード（{@link ObservationSchema#encodeFu}。役満は 0）。 */
  public int winFuCode(int win) {
    return win(win, WinBits.FU_CODE);
  }

  /** 和了の役満の倍率。 */
  public int winYakuman(int win) {
    return win(win, WinBits.YAKUMAN);
  }

  public boolean winPao(int win) {
    return win(win, WinBits.PAO) != 0;
  }

  /** 和了で裏ドラが乗りうる（立直している）なら true。 */
  public boolean winUraEligible(int win) {
    return win(win, WinBits.URA_ELIGIBLE) != 0;
  }

  /** 和了でめくりうる裏ドラ表示牌の数（今めくってあるドラ表示牌の数と同じ。立直していない和了でもこの数を書く）。 */
  public int winUraIndicators(int win) {
    return win(win, WinBits.URA_INDICATORS);
  }

  /** 和了の後の順位を付けるときに置いた、精算の仮定（ツモなら精算は 1 通りに決まる。ロンは単独の和了とみなす）。 */
  public SettlementAssumption winSettlement(int win) {
    return schema.settlements.get(win(win, WinBits.SETTLEMENT));
  }

  /** 和了の後の、指定した相対席の順位（0 始まり。同点は起家に近い席が上）。 */
  public int winRank(int win, int seat) {
    return schema.win.get(words, start(Section.WIN), checkWin(win), WinBits.RANK, seat);
  }

  /** 和了による、指定した相対席の点の移動（受け取りが正）。裏ドラと場の供託は含まず、ロンでは本場も含まない。 */
  public int winPayment(int win, int seat) {
    return ObservationSchema.decodePoints(
        schema.win.get(words, start(Section.WIN), checkWin(win), WinBits.PAYMENT, seat));
  }

  long[] words() {
    return words;
  }

  int base() {
    return base;
  }

  /** セクションが配列の中で始まる位置。 */
  private int start(Section section) {
    return base + start[section.ordinal()];
  }

  private int header(HeaderBits field) {
    return schema.header.get(words, base, 0, field);
  }

  private int seat(int seat, SeatBits field) {
    return schema.seat.get(words, start(Section.SEAT), Objects.checkIndex(seat, seats), field);
  }

  private int discard(int event, DiscardBits field) {
    long word = eventWord(event);
    requireKind(event, word, EventKind.DISCARD);
    return schema.discard.get(word, field);
  }

  /** イベントの記録を入れた {@code long}。番号の範囲を確かめてから読む。 */
  private long eventWord(int event) {
    return schema.event.record(words, start(Section.EVENT), Objects.checkIndex(event, events));
  }

  private int win(int win, WinBits field) {
    return schema.win.get(words, start(Section.WIN), checkWin(win), field);
  }

  private int checkWin(int win) {
    return Objects.checkIndex(win, wins);
  }

  private void requireKind(int event, long word, EventKind kind) {
    if (schema.eventKinds.get(schema.event.get(word, EventBits.KIND)) != kind) {
      throw new IllegalStateException("Public event " + event + " is not a " + kind + " event");
    }
  }

  /**
   * 全ての合法手について、同値な打牌の代表を決める。手を番号の小さい順に見て、組の中で最初に出た手を代表にする。組は打牌の種類・
   * 牌・赤の選び方で決まるので、組ごとに最初の手を覚えておけば 1 回の走査で決まる。
   */
  private void findRepresentatives() {
    if (++twinStamp == 0) {
      Arrays.fill(twinSeen, 0);
      twinStamp = 1;
    }
    for (int action = 0; action < actions; action++) {
      int key = twinKey(legal[action]);
      int found = action;
      if (key >= 0) {
        if (twinSeen[key] == twinStamp) {
          found = twinFirst[key];
        } else {
          twinSeen[key] = twinStamp;
          twinFirst[key] = action;
        }
      }
      representatives[action] = found;
    }
    representativesFound = true;
  }

  /** 同値な打牌の組の番号（{@link #twins} が true になる手どうしで同じ値）。打牌でない手は −1。 */
  private static int twinKey(Action action) {
    int type =
        switch (action.type()) {
          case DAHAI -> 0;
          case RIICHI_DAHAI -> 1;
          default -> -1;
        };
    if (type < 0) {
      return -1;
    }
    return (type * Tiles.KINDS + action.tile()) * 2 + (action.selection().red() ? 1 : 0);
  }

  /**
   * イベントを前から 1 回だけ読み、打牌した席・副露の有無・抜いた北の数を覚える（付け替えると読み直す）。北抜きの無いルールでは、
   * 全ての席が打牌し、副露も見つかった所で読むのをやめる。
   */
  private void scanEvents() {
    if (scanned) {
      return;
    }
    int allSeats = variant.seatMask();
    int discarded = 0;
    boolean melds = false;
    boolean countNuki = nukiKind >= 0;
    Arrays.fill(nuki, 0);
    int at = start(Section.EVENT);
    for (int event = 0; event < events && (discarded != allSeats || !melds || countNuki); event++) {
      long word = schema.event.record(words, at, event);
      int kind = schema.event.get(word, EventBits.KIND);
      if (kind == discardKind) {
        discarded |= 1 << schema.event.get(word, EventBits.AUTHOR);
      } else if (kind == meldKind) {
        melds = true;
      } else {
        nuki[schema.event.get(word, EventBits.AUTHOR)]++;
      }
    }
    discardedSeats = discarded;
    meldMade = melds;
    scanned = true;
  }

  private boolean ownDraw() {
    return lastEvent() == LastEvent.DRAW && lastEventSeat() == 0;
  }

  /**
   * {@link #readEvent} で読んだ、1 つのイベントの値（種類、起こした席、種類ごとのフィールド）。呼び出し側が 1 つ持って使い回す。
   * 種類の違うイベントのフィールドを読むと、フィールドごとのメソッドと同じく IllegalStateException を投げる。
   */
  public static final class EventRecord {

    private int event = -1;
    private EventKind kind;
    private int author;
    private int tile;
    private boolean aka;
    private boolean tsumogiri;
    private boolean riichi;
    private int calledBy;
    private Meld meld;

    /** 読んだイベントの番号（まだ読んでいなければ −1）。 */
    public int event() {
      return event;
    }

    public EventKind kind() {
      return kind;
    }

    /** イベントを起こした席（相対席）。 */
    public int author() {
      return author;
    }

    public int discardTile() {
      requireKind(EventKind.DISCARD);
      return tile;
    }

    public boolean discardAka() {
      requireKind(EventKind.DISCARD);
      return aka;
    }

    public boolean discardTsumogiri() {
      requireKind(EventKind.DISCARD);
      return tsumogiri;
    }

    /** 打った牌が立直の宣言牌なら true。 */
    public boolean discardRiichi() {
      requireKind(EventKind.DISCARD);
      return riichi;
    }

    /** 打牌を鳴いた席（相対席。鳴かれていなければ −1）。 */
    public int discardCalledBy() {
      requireKind(EventKind.DISCARD);
      return calledBy;
    }

    /** 副露・加槓のイベントの時点での面子。 */
    public Meld meld() {
      requireKind(EventKind.MELD);
      return meld;
    }

    private void requireKind(EventKind expected) {
      if (kind != expected) {
        throw new IllegalStateException("Public event " + event + " is not a " + expected + " event");
      }
    }
  }
}
