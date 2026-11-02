package com.epsilon.prism.observe;

import static com.epsilon.prism.observe.ObservationSchema.CALL_DISCARD;
import static com.epsilon.prism.observe.ObservationSchema.DISCARD;
import static com.epsilon.prism.observe.ObservationSchema.EVENT;
import static com.epsilon.prism.observe.ObservationSchema.HAND;
import static com.epsilon.prism.observe.ObservationSchema.HEADER;
import static com.epsilon.prism.observe.ObservationSchema.LEGAL;
import static com.epsilon.prism.observe.ObservationSchema.MELD;
import static com.epsilon.prism.observe.ObservationSchema.SEAT;
import static com.epsilon.prism.observe.ObservationSchema.WIN;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.prism.Limits;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.observe.ObservationSchema.CallDiscardBits;
import com.epsilon.prism.observe.ObservationSchema.DiscardBits;
import com.epsilon.prism.observe.ObservationSchema.EventBits;
import com.epsilon.prism.observe.ObservationSchema.EventKind;
import com.epsilon.prism.observe.ObservationSchema.HandBits;
import com.epsilon.prism.observe.ObservationSchema.HeaderBits;
import com.epsilon.prism.observe.ObservationSchema.LastEvent;
import com.epsilon.prism.observe.ObservationSchema.LegalBits;
import com.epsilon.prism.observe.ObservationSchema.MeldBits;
import com.epsilon.prism.observe.ObservationSchema.SeatBits;
import com.epsilon.prism.observe.ObservationSchema.Section;
import com.epsilon.prism.observe.ObservationSchema.WinBits;
import java.util.Objects;

/**
 * 記録 O（{@link ObservationSchema}）の読み手。{@link #bind} で別の O に付け替えて使い回し、読むときに割り当てをしない（flyweight）。
 * 付け替えるので、読むスレッドごとに 1 つ持つ。
 *
 * <p>席は観測席から見た相対席（0 自分、1 下家、2 対面、3 上家）で返す。なしは牌・席・番号で −1、enum で null。手は
 * {@link Action#fromIndex}、面子は {@link Meld#fromCanonicalId} の共有の表から返す。
 *
 * <p>O に書かない値のうち、使う側が同じ答えを要るものもここで出す。場風は局から、初巡・鳴きの有無・第一打の前は出来事から、ツモ牌は
 * 直前の出来事から出す。双子の代表・遷移の数・槓ドラの表示の位置の決まりもここに置き、書き手は {@link #countTransitions} で頭の遷移の
 * 数を書き、Tokenizer は {@link #representative}・{@link #transitionsOf} で同じ遷移を並べる。
 *
 * <p>読む値の番号（出来事・合法手など）は範囲を確かめる。種類ごとの欄（打牌・副露）は、その種類の出来事でなければ例外にする。出来事の全ての
 * 欄を読む側は、欄ごとの口ではなく {@link #readEvent} で記録を 1 回だけ読む。
 */
public final class Observation {

  private static final EngineDecisionKind[] DECISION_KINDS = EngineDecisionKind.values();
  private static final RiichiState[] RIICHI_STATES = RiichiState.values();
  private static final LastEvent[] LAST_EVENTS = LastEvent.values();
  private static final TurnEvent.KanKind[] KAN_KINDS = TurnEvent.KanKind.values();
  private static final EventKind[] EVENT_KINDS = EventKind.values();
  private static final SettlementAssumption[] SETTLEMENTS = SettlementAssumption.values();
  private static final int DISCARD_KIND = EventKind.DISCARD.ordinal();
  private static final int ALL_SEATS = (1 << GameState.NUM_PLAYERS) - 1;

  private final int[] start = new int[ObservationSchema.SECTIONS.length];
  private final Action[] legal = new Action[Limits.MAX_LEGAL_ACTIONS];
  private final int[] representatives = new int[Limits.MAX_LEGAL_ACTIONS];
  private boolean representativesFound;
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

  /**
   * 読む O を付け替える。
   *
   * @param words O を置いた配列
   * @param base O の先頭
   * @return この読み手
   * @throws IndexOutOfBoundsException O が配列に収まらないとき
   */
  public Observation bind(long[] words, int base) {
    this.words = Objects.requireNonNull(words, "words");
    this.base = base;
    longs = ObservationSchema.place(words, base, start);
    Objects.checkFromIndexSize(base, longs, words.length);
    events = header(HeaderBits.EVENTS);
    actions = header(HeaderBits.LEGAL_ACTIONS);
    callDiscards = header(HeaderBits.CALL_DISCARDS);
    wins = header(HeaderBits.WINS);
    int legalStart = start(Section.LEGAL);
    for (int action = 0; action < actions; action++) {
      legal[action] = Action.fromIndex(LEGAL.get(words, legalStart, action, LegalBits.ACTION));
    }
    scanned = false;
    representativesFound = false;
    return this;
  }

  /** O の {@code long} の数。 */
  public int longs() {
    return longs;
  }

  public EngineDecisionKind decisionKind() {
    return DECISION_KINDS[header(HeaderBits.DECISION_KIND)];
  }

  /** 観測席（絶対席）。 */
  public int observer() {
    return header(HeaderBits.OBSERVER);
  }

  /** 相対席を絶対席に戻す（同点の順位の席順など）。 */
  public int absoluteSeat(int seat) {
    return (observer() + Objects.checkIndex(seat, GameState.NUM_PLAYERS)) % GameState.NUM_PLAYERS;
  }

  /** 親の相対席。 */
  public int dealer() {
    return header(HeaderBits.DEALER);
  }

  /** 局（東 1 局 0 … 西 4 局 11）。 */
  public int kyoku() {
    return header(HeaderBits.KYOKU);
  }

  /** 場風の牌（{@link Tile#TON} … {@link Tile#PEI}）。局から出す（4 局ごとに次の風）。 */
  public int roundWindTile() {
    return Tile.TON + kyoku() / GameState.NUM_PLAYERS;
  }

  public int honba() {
    return header(HeaderBits.HONBA);
  }

  /** 供託の本数。 */
  public int kyotaku() {
    return header(HeaderBits.KYOTAKU);
  }

  /** 王牌を除く山の残り。 */
  public int remainingWall() {
    return header(HeaderBits.WALL);
  }

  /** 第一巡の途中（打牌の出来事のない席がある）。 */
  public boolean firstCycle() {
    scanEvents();
    return discardedSeats != ALL_SEATS;
  }

  /** この局で副露（暗槓を含む）があった（副露の出来事がある）。 */
  public boolean callMade() {
    scanEvents();
    return meldMade;
  }

  /** 自分の見逃しの振聴（同巡の振聴と立直後の振聴を兼ねる。捨て牌の振聴は含まない）。 */
  public boolean furitenTemporary() {
    return header(HeaderBits.FURITEN_TEMPORARY) != 0;
  }

  /** 直前の出来事（なしは null）。 */
  public LastEvent lastEvent() {
    return optional(LAST_EVENTS, header(HeaderBits.LAST_EVENT));
  }

  /** 直前の出来事の相対席（なしは −1）。 */
  public int lastEventSeat() {
    return header(HeaderBits.LAST_EVENT_SEAT);
  }

  /** 直前の出来事の牌（なしと他家のツモは −1）。 */
  public int lastEventTile() {
    return header(HeaderBits.LAST_EVENT_TILE);
  }

  /** 直前の出来事の牌が赤。 */
  public boolean lastEventAka() {
    return header(HeaderBits.LAST_EVENT_AKA) != 0;
  }

  /** 直前のツモが嶺上。 */
  public boolean lastEventRinshan() {
    return header(HeaderBits.LAST_EVENT_RINSHAN) != 0;
  }

  /** 直前の槓の宣言の種類（直前の出来事が槓の宣言でなければ null）。 */
  public TurnEvent.KanKind lastKanKind() {
    return optional(KAN_KINDS, header(HeaderBits.LAST_KAN_KIND));
  }

  /** 槓ドラの表示が後に延びている。 */
  public boolean deferredDora() {
    return header(HeaderBits.DEFERRED_DORA) != 0;
  }

  /** 今ツモった牌（直前の出来事が自分のツモでなければ −1）。直前の出来事の牌と同じ事実。 */
  public int drawnTile() {
    return ownDraw() ? lastEventTile() : -1;
  }

  /** 今ツモった牌が赤。 */
  public boolean drawnAka() {
    return ownDraw() && lastEventAka();
  }

  /** 自分の手（副露を除き、ツモ牌を含む）の色ごとの赤（AkaTileMask：bit 0 萬、1 筒、2 索）。 */
  public int handAkaMask() {
    return header(HeaderBits.HAND_AKA);
  }

  /** ドラ表示の数（初めの表示を含むので 1 以上）。 */
  public int indicators() {
    return header(HeaderBits.INDICATORS);
  }

  /** 槓ドラの表示の数（初めの表示を除いたドラ表示の数）。 */
  public int kanDoraIndicators() {
    return indicators() - 1;
  }

  /** 表示の順の {@code index} 番目のドラ表示の牌。 */
  public int indicatorTile(int index) {
    return HEADER.get(
        words, base, 0, HeaderBits.INDICATOR, Objects.checkIndex(index, indicators()));
  }

  /** ドラ表示に出た色ごとの赤（AkaTileMask）。どの表示が赤かは O から決まらない（エンジンが色ごとにしか持たない）。 */
  public int indicatorAkaMask() {
    return header(HeaderBits.INDICATOR_AKA);
  }

  /** O の出来事の数（打牌・副露・加槓。槓ドラの表示は含まない）。 */
  public int events() {
    return events;
  }

  /**
   * 公開の出来事の数（O の出来事 ＋ 槓ドラの表示）を返す。入力の出来事トークンはこの数だけ並べ、初めのドラ表示は出来事にしない。O の出来事と
   * 表示の数の範囲から、{@link Limits#MAX_EVENT_TOKENS} を超えない。
   *
   * @return 公開の出来事の数
   */
  public int publicEvents() {
    return events + kanDoraIndicators();
  }

  public int legalActions() {
    return actions;
  }

  /** 遷移の数（書き手が {@link #countTransitions} で書いた値）。 */
  public int transitions() {
    return header(HeaderBits.TRANSITIONS);
  }

  /** チー・ポンの後の打牌の数（全ての合法手の和）。 */
  public int callDiscards() {
    return callDiscards;
  }

  /** 即時の和了の数。 */
  public int wins() {
    return wins;
  }

  /** 相対席 {@code seat} の点。 */
  public int score(int seat) {
    return ObservationSchema.decodePoints(seat(seat, SeatBits.SCORE));
  }

  /** 相対席 {@code seat} の立直の状態。 */
  public RiichiState riichi(int seat) {
    return RIICHI_STATES[seat(seat, SeatBits.RIICHI)];
  }

  /** 相対席 {@code seat} に一発がある。 */
  public boolean ippatsu(int seat) {
    return seat(seat, SeatBits.IPPATSU) != 0;
  }

  /** 相対席 {@code seat} は第一打の前（その席の打牌の出来事がない）。 */
  public boolean beforeFirstDiscard(int seat) {
    Objects.checkIndex(seat, GameState.NUM_PLAYERS);
    scanEvents();
    return (discardedSeats & 1 << seat) == 0;
  }

  /** 自分の手（副露を除き、ツモ牌を含む）の牌 {@code tile} の枚数。 */
  public int handCount(int tile) {
    return HAND.get(words, start(Section.HAND), 0, HandBits.COUNT, tile);
  }

  /**
   * 自分の手（副露を除き、ツモ牌を含む）の全ての牌の枚数を、牌の番号の順に読む。
   *
   * @param counts 枚数を入れる配列（先頭から {@link Tile#NUM_TILE_TYPES} 個）
   */
  public void handCounts(int[] counts) {
    HAND.getAll(words, start(Section.HAND), 0, HandBits.COUNT, counts);
  }

  /** 出来事 {@code event}（番号の順）の種類。 */
  public EventKind eventKind(int event) {
    return EVENT_KINDS[EVENT.get(eventWord(event), EventBits.KIND)];
  }

  /** 出来事の作者の相対席。 */
  public int eventAuthor(int event) {
    return EVENT.get(eventWord(event), EventBits.AUTHOR);
  }

  /**
   * 出来事 {@code event} の記録を 1 回だけ読み、種類・作者と種類ごとの欄を {@code out} に入れる。出来事を全て読む側（入力の派生）は、
   * 欄ごとの口の代わりにこれを使う。
   *
   * @param event 出来事の番号
   * @param out 入れる先（呼び手が持ち、使い回す）
   * @return {@code out}
   */
  public EventRecord readEvent(int event, EventRecord out) {
    long word = eventWord(event);
    out.event = event;
    out.kind = EVENT_KINDS[EVENT.get(word, EventBits.KIND)];
    out.author = EVENT.get(word, EventBits.AUTHOR);
    if (out.kind == EventKind.DISCARD) {
      out.tile = DISCARD.get(word, DiscardBits.TILE);
      out.aka = DISCARD.get(word, DiscardBits.AKA) != 0;
      out.tsumogiri = DISCARD.get(word, DiscardBits.TSUMOGIRI) != 0;
      out.riichi = DISCARD.get(word, DiscardBits.RIICHI) != 0;
      out.calledBy = DISCARD.get(word, DiscardBits.CALLED_BY);
      out.meld = null;
    } else {
      out.meld = Meld.fromCanonicalId(MELD.get(word, MeldBits.MELD));
    }
    return out;
  }

  /** 打牌の出来事の牌。 */
  public int discardTile(int event) {
    return discard(event, DiscardBits.TILE);
  }

  /** 打牌の出来事の牌が赤。 */
  public boolean discardAka(int event) {
    return discard(event, DiscardBits.AKA) != 0;
  }

  /** 打牌の出来事がツモ切り。 */
  public boolean discardTsumogiri(int event) {
    return discard(event, DiscardBits.TSUMOGIRI) != 0;
  }

  /** 打牌の出来事が立直の宣言牌。 */
  public boolean discardRiichi(int event) {
    return discard(event, DiscardBits.RIICHI) != 0;
  }

  /** 打牌の出来事を鳴いた相対席（鳴かれていなければ −1）。鳴いた副露は {@code event + 1} の出来事。 */
  public int discardCalledBy(int event) {
    return discard(event, DiscardBits.CALLED_BY);
  }

  /**
   * 副露・加槓の出来事の時点の面子を返す（後に加槓したポンも、ポンの出来事ではポン）。
   *
   * @param event 出来事の番号
   * @return 面子
   * @throws IllegalStateException 副露・加槓の出来事でないとき
   */
  public Meld meld(int event) {
    long word = eventWord(event);
    requireKind(event, word, EventKind.MELD);
    return Meld.fromCanonicalId(MELD.get(word, MeldBits.MELD));
  }

  /**
   * 槓ドラの表示を置く出来事の番号を返す。{@code kan} 番目の槓（出来事の順）のドラ表示（表示の {@code kan + 1} 番目）は、暗槓なら
   * 槓の出来事の直後、大明槓・加槓なら作者の次の出来事（打牌か次の槓）の直後に開く（エンジンの規則）。
   *
   * @param kan 開いた槓ドラの番号（0 始まり、{@link #kanDoraIndicators()} 未満）
   * @return その表示の直前の出来事の番号
   * @throws IllegalStateException 規則の位置の出来事がないとき（O が規則と合わない）
   */
  public int kanDoraEvent(int kan) {
    Objects.checkIndex(kan, kanDoraIndicators());
    int seen = 0;
    for (int event = 0; event < events; event++) {
      if (eventKind(event) != EventKind.MELD) {
        continue;
      }
      Meld meld = meld(event);
      if (!meld.isKan() || seen++ < kan) {
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
      throw new IllegalStateException("kan dora " + kan + " is shown before its author acts again");
    }
    throw new IllegalStateException("kan dora " + kan + " is shown without its kan");
  }

  public Action action(int action) {
    return legal[Objects.checkIndex(action, actions)];
  }

  /** 合法手 {@code action} の後の打牌の数（チー・ポンだけ。ほかは 0）。 */
  public int callDiscardsOf(int action) {
    return LEGAL.get(
        words, start(Section.LEGAL), Objects.checkIndex(action, actions), LegalBits.CALL_DISCARDS);
  }

  /** チー・ポンの後の打牌の {@code index} 番目（チー・ポンの合法手の順にまとめた番号）。 */
  public Action callDiscard(int index) {
    return Action.fromIndex(
        CALL_DISCARD.get(
            words,
            start(Section.CALL_DISCARD),
            Objects.checkIndex(index, callDiscards),
            CallDiscardBits.ACTION));
  }

  /**
   * 合法手 {@code action} の双子の代表を返す。双子は種類（DAHAI か RIICHI_DAHAI）・牌・赤の選びが同じ打牌（{@link Twins} の
   * 決まり）で、番号の最も小さい手を代表にする。代表は付け替えの後に初めて聞かれたときに全ての手の分を 1 回だけ決める。
   *
   * @param action 合法手の番号
   * @return 代表の合法手の番号（代表なら自分）
   */
  public int representative(int action) {
    Objects.checkIndex(action, actions);
    if (!representativesFound) {
      findRepresentatives();
    }
    return representatives[action];
  }

  /**
   * 合法手 {@code action} の遷移の数を返す。代表でない双子は 0（代表の遷移を使う）、チー・ポンは後の打牌ごとに 1 つ、ほかは 1。
   *
   * @param action 合法手の番号
   * @return 遷移の数
   */
  public int transitionsOf(int action) {
    if (representative(action) != action) {
      return 0;
    }
    return ObservationSchema.hasCallDiscards(action(action).type()) ? callDiscardsOf(action) : 1;
  }

  /** 遷移の総数（{@link #transitionsOf} の和）。書き手が頭の遷移の数に書く値。 */
  public int countTransitions() {
    int total = 0;
    for (int action = 0; action < actions; action++) {
      total += transitionsOf(action);
    }
    return total;
  }

  /** ロン・ツモの合法手 {@code action} の和了の番号（ロン・ツモでなければ −1）。 */
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

  /** 和了 {@code win} の翻（裏なし、役満は 0）。 */
  public int winHan(int win) {
    return win(win, WinBits.HAN);
  }

  /** 和了の符の符号（{@link ObservationSchema#encodeFu}。役満は 0）。 */
  public int winFuCode(int win) {
    return win(win, WinBits.FU_CODE);
  }

  /** 和了の役満の倍率。 */
  public int winYakuman(int win) {
    return win(win, WinBits.YAKUMAN);
  }

  /** 和了に包が成り立つ。 */
  public boolean winPao(int win) {
    return win(win, WinBits.PAO) != 0;
  }

  /** 和了に裏ドラの資格がある（立直している）。 */
  public boolean winUraEligible(int win) {
    return win(win, WinBits.URA_ELIGIBLE) != 0;
  }

  /** 和了で開きうる裏ドラ表示の数。 */
  public int winUraIndicators(int win) {
    return win(win, WinBits.URA_INDICATORS);
  }

  /** 和了の後の順位を出した精算の仮定（ツモは一意、ロンは単独の和了）。 */
  public SettlementAssumption winSettlement(int win) {
    return SETTLEMENTS[win(win, WinBits.SETTLEMENT)];
  }

  /** 和了の後の相対席 {@code seat} の順位（0 始まり、同点は席順）。 */
  public int winRank(int win, int seat) {
    return WIN.get(words, start(Section.WIN), checkWin(win), WinBits.RANK, seat);
  }

  /** 和了の相対席 {@code seat} の点の移動（受け取りは正、裏なし、ロンは本場なし）。 */
  public int winPayment(int win, int seat) {
    return ObservationSchema.decodePoints(
        WIN.get(words, start(Section.WIN), checkWin(win), WinBits.PAYMENT, seat));
  }

  /** 節の先頭（配列の添字）。 */
  private int start(Section section) {
    return base + start[section.ordinal()];
  }

  private int header(HeaderBits field) {
    return HEADER.get(words, base, 0, field);
  }

  private int seat(int seat, SeatBits field) {
    return SEAT.get(
        words, start(Section.SEAT), Objects.checkIndex(seat, GameState.NUM_PLAYERS), field);
  }

  private int discard(int event, DiscardBits field) {
    long word = eventWord(event);
    requireKind(event, word, EventKind.DISCARD);
    return DISCARD.get(word, field);
  }

  /** 出来事の記録の語（番号の範囲を確かめる）。 */
  private long eventWord(int event) {
    return EVENT.record(words, start(Section.EVENT), checkEvent(event));
  }

  private int win(int win, WinBits field) {
    return WIN.get(words, start(Section.WIN), checkWin(win), field);
  }

  private int checkEvent(int event) {
    return Objects.checkIndex(event, events);
  }

  private int checkWin(int win) {
    return Objects.checkIndex(win, wins);
  }

  private static void requireKind(int event, long word, EventKind kind) {
    if (EVENT_KINDS[EVENT.get(word, EventBits.KIND)] != kind) {
      throw new IllegalStateException("public event " + event + " is not a " + kind + " event");
    }
  }

  /** 全ての合法手の双子の代表を決める（番号の小さい順に探し、最初に見つかった同じ打牌を代表にする）。 */
  private void findRepresentatives() {
    for (int action = 0; action < actions; action++) {
      Action self = legal[action];
      int found = action;
      if (Twins.discard(self)) {
        int tile = self.tileType();
        for (int other = 0; other < action; other++) {
          Action candidate = legal[other];
          if (candidate.tileType() == tile && Twins.same(self, candidate)) {
            found = other;
            break;
          }
        }
      }
      representatives[action] = found;
    }
    representativesFound = true;
  }

  /** 出来事を 1 回だけ走査し、打牌のある席と副露の有無を持つ（付け替えると読み直す）。 */
  private void scanEvents() {
    if (scanned) {
      return;
    }
    int seats = 0;
    boolean melds = false;
    int at = start(Section.EVENT);
    for (int event = 0; event < events && (seats != ALL_SEATS || !melds); event++) {
      long word = EVENT.record(words, at, event);
      if (EVENT.get(word, EventBits.KIND) == DISCARD_KIND) {
        seats |= 1 << EVENT.get(word, EventBits.AUTHOR);
      } else {
        melds = true;
      }
    }
    discardedSeats = seats;
    meldMade = melds;
    scanned = true;
  }

  private boolean ownDraw() {
    return lastEvent() == LastEvent.DRAW && lastEventSeat() == 0;
  }

  /** ordinal（なしは −1）を enum の値（なしは null）にする。 */
  private static <T extends Enum<T>> T optional(T[] values, int ordinal) {
    return ordinal < 0 ? null : values[ordinal];
  }

  /**
   * {@link #readEvent} で読んだ出来事 1 つの値（種類・作者と、種類ごとの欄）。呼び手が 1 つ持って使い回すので、読むときに割り当てをしない。
   * 種類ごとの欄は、その種類の出来事でなければ例外にする（欄ごとの口と同じ決まり）。
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

    /** 読んだ出来事の番号（まだ読んでいなければ −1）。 */
    public int event() {
      return event;
    }

    public EventKind kind() {
      return kind;
    }

    /** 作者の相対席。 */
    public int author() {
      return author;
    }

    /** 打牌の牌。 */
    public int discardTile() {
      requireKind(EventKind.DISCARD);
      return tile;
    }

    /** 打牌の牌が赤。 */
    public boolean discardAka() {
      requireKind(EventKind.DISCARD);
      return aka;
    }

    /** 打牌がツモ切り。 */
    public boolean discardTsumogiri() {
      requireKind(EventKind.DISCARD);
      return tsumogiri;
    }

    /** 打牌が立直の宣言牌。 */
    public boolean discardRiichi() {
      requireKind(EventKind.DISCARD);
      return riichi;
    }

    /** 打牌を鳴いた相対席（鳴かれていなければ −1）。 */
    public int discardCalledBy() {
      requireKind(EventKind.DISCARD);
      return calledBy;
    }

    /** 副露・加槓の出来事の時点の面子。 */
    public Meld meld() {
      requireKind(EventKind.MELD);
      return meld;
    }

    private void requireKind(EventKind expected) {
      if (kind != expected) {
        throw new IllegalStateException("public event " + event + " is not a " + expected + " event");
      }
    }
  }
}
