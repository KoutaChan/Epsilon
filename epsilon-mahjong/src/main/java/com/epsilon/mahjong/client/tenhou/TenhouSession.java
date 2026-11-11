package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.client.LiveTable;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.io.tenhou.TenhouMeldCode;
import com.epsilon.mahjong.io.tenhou.TenhouRound;
import com.epsilon.mahjong.io.tenhou.TenhouTag;
import com.epsilon.mahjong.io.tenhou.TenhouTiles;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 天鳳での 1 半荘の対局（四人打ちか三人打ち）。サーバのメッセージを受けて LiveTable を進め、自分の判断は Decider に選ばせて、
 * 返すメッセージを作る。
 *
 * <p>サーバは自分を席 0 とする相対席で書いてくる。INIT の親から自分の絶対席を決め（TenhouRound）、席を絶対席に直して卓に渡す。
 * 選べる手は、エンジンの合法手のうちサーバが t のビットで出した手だけにする（打牌・暗槓・加槓はいつも出せる）。
 *
 * <p>自分のツモの後と、自分のチー・ポンの後は必ず手を選ぶ。他家の打牌・加槓には、t があるときだけ答える。t があっても見送りのほかに
 * 出せる手が無ければ、見送りを返す。t とエンジンの合法手が食い違えば警告を出す。自分の打牌や鳴きで送る物理牌は、卓がそのイベントを
 * 受けたときに手から除く牌にする（{@code LiveTable.tiles}）。
 *
 * <p>三人打ちの卓では、N の属性 m にある鳴き元を、天鳳の 4 席（自分が席 0、席 3 は空き席）で鳴いた席から数えた値として読む。
 * m が北抜きのコードなら北抜きとして読む。北抜きは打牌・暗槓と同じく、t のビットが無くても出せる手とする。自分の北抜きは手番の
 * 判断で返し、続く補充のツモで次の判断をする。他家の北抜きには、t があるときだけ答える（抜いた北へのロン）。北抜きは N の type 10
 * で返す。この返事は牌を指定しないので、どの北を抜いたかはサーバが N の m で知らせる。自分の手からは、その北を除く。
 *
 * <p>局の結果（AGARI・RYUUKYOKU）は RoundResult にして、次の INIT か終局（owari）で卓の精算と照らし合わせる。読めない結果は
 * 照らし合わせずに、合わなかった局に数える。対局の始まりと局の終わりには、NEXTREADY を 1 回だけ返す。1 つのスレッドからだけ
 * 使う。
 */
public final class TenhouSession {

  private static final Logger log = LoggerFactory.getLogger(TenhouSession.class);

  /** いまの局の結果を読めたか。 */
  private enum Result {
    /** AGARI も RYUUKYOKU もまだ来ていない。 */
    NONE,
    /** 読めた。次の INIT か終局で照らし合わせる。 */
    READ,
    /** 読めなかった。照らし合わせない。 */
    UNREADABLE
  }

  private final Variant variant;
  private final Decider decider;
  private final TenhouRound round;

  /** 返す手で出す物理牌。{@code LiveTable.tiles} が書く。 */
  private final int[] tiles = new int[Tiles.COPIES];

  /** 最初の INIT で作る。 */
  private LiveTable live;

  private Result result = Result.NONE;

  /** いまの区切り（対局の始まりか局の終わり）で、もう NEXTREADY を返したか。 */
  private boolean ready;

  /** 終局の結果を受けたか。 */
  private boolean ended;

  /** 局の結果が読めなかった局の数。 */
  private int unreadable;

  /**
   * @param variant 申し込んだ卓のルール。GO の卓の種別がこのルールと違えば、その GO で IllegalArgumentException を投げる
   * @param decider 手を選ぶもの
   */
  public TenhouSession(Variant variant, Decider decider) {
    this.variant = Objects.requireNonNull(variant, "variant");
    this.decider = Objects.requireNonNull(decider, "decider");
    round = new TenhouRound(variant);
  }

  /** 終局の結果を受けたなら true。 */
  public boolean ended() {
    return ended;
  }

  /** 局の結果が卓の精算と合わなかった局と、読めなかった局の数。 */
  public int mismatches() {
    return unreadable + (live == null ? 0 : live.mismatches());
  }

  /**
   * サーバのメッセージを 1 つ受ける。知らないタグは読み飛ばす。
   *
   * @param message サーバのメッセージ
   * @return 返すメッセージ。無ければ空
   * @throws IllegalArgumentException 読めないメッセージのときと、GO の卓の種別が違うとき
   * @throws IllegalStateException 卓の進みと合わないメッセージのとき
   */
  public List<String> handle(String message) {
    TenhouTag tag = TenhouCodec.parse(message);
    return switch (tag.name()) {
      case "GO" -> {
        Variant table = TenhouXml.variant(tag.integer("type"));
        if (table != variant) {
          throw new IllegalArgumentException(
              "Tenhou type " + tag.get("type") + " is a " + table + " table, not " + variant);
        }
        yield List.of(TenhouCodec.gok());
      }
      case "REJOIN" -> rejoin(tag);
      case "UN" -> {
        players(tag);
        yield List.of();
      }
      case "TAIKYOKU" -> ready();
      case "INIT" -> {
        init(tag);
        yield List.of();
      }
      case "N" -> meld(tag);
      case "REACH" -> {
        riichi(tag);
        yield List.of();
      }
      case "DORA" -> {
        table().dora(TenhouTiles.tile(tag.integer("hai"), variant.tiles()));
        yield List.of();
      }
      case "AGARI", "RYUUKYOKU" -> roundEnd(tag);
      default -> move(tag);
    };
  }

  private List<String> rejoin(TenhouTag tag) {
    String target = tag.get("t");
    if (target == null || target.isEmpty()) {
      log.warn("Tenhou asks to join again without a target");
      return List.of();
    }
    return List.of(TenhouCodec.join(target));
  }

  /** 打ち手の名前をログに出す。名前は自分を 0 とする相対席の順に並び、URL エンコードされている。 */
  private void players(TenhouTag tag) {
    for (int seat = 0; seat < variant.seats(); seat++) {
      String name = tag.get("n" + seat);
      if (name != null && !name.isEmpty()) {
        log.info("Tenhou player n{}: {}", seat, URLDecoder.decode(name, StandardCharsets.UTF_8));
      }
    }
  }

  private List<String> ready() {
    if (ready) {
      return List.of();
    }
    ready = true;
    return List.of(TenhouCodec.nextReady());
  }

  private void init(TenhouTag tag) {
    RoundResult previous = result == Result.READ ? result() : null;
    RoundStart start = round.start(tag);
    if (previous != null) {
      live.checkRound(previous, start);
    }
    int seat = round.rotation();
    if (live == null) {
      live = new LiveTable(variant, seat, decider);
    } else if (live.seat() != seat) {
      throw new IllegalStateException("Tenhou moves this player from seat " + live.seat() + " to seat " + seat);
    }
    live.startRound(start, TenhouTiles.tiles(tag.required("hai"), variant.tiles()), round.indicator());
    result = Result.NONE;
    ready = false;
    log.info("Tenhou {} starts with this player at seat {}", start, seat);
  }

  /** ツモか打牌のタグを受ける。ほかの知らないタグは読み飛ばす。 */
  private List<String> move(TenhouTag tag) {
    TenhouCodec.Move move = TenhouCodec.move(tag.name(), variant);
    if (move == null) {
      log.debug("Tenhou message {} is not read", tag.name());
      return List.of();
    }
    LiveTable table = table();
    int seat = round.seat(move.seat());
    boolean own = seat == table.seat();
    return switch (move.kind()) {
      case DRAW -> {
        if (!own) {
          table.draw(seat, TurnEvent.NONE);
          yield List.of();
        }
        if (move.tile() == TurnEvent.NONE) {
          throw new IllegalArgumentException("The own draw " + tag.name() + " names no tile");
        }
        table.draw(seat, move.tile());
        yield decide(TenhouCodec.offers(tag), true);
      }
      case DISCARD -> {
        table.discard(seat, move.tile(), move.tsumogiri() || own && move.tile() == table.drawn());
        int offers = TenhouCodec.offers(tag);
        yield own || offers == 0 ? List.of() : decide(offers, false);
      }
    };
  }

  /** チー・ポン・大明槓・加槓・暗槓か、三人打ちの卓の北抜きを受ける。 */
  private List<String> meld(TenhouTag tag) {
    LiveTable table = table();
    int seat = round.seat(tag.integer("who"));
    int code = tag.integer("m");
    if (TenhouMeldCode.nuki(code)) {
      table.nuki(seat, TenhouMeldCode.nukiTile(code, variant));
      int offers = TenhouCodec.offers(tag);
      return seat != table.seat() && offers != 0 ? decide(offers, false) : List.of();
    }
    Meld meld = TenhouMeldCode.decode(code, variant, tag.integer("who")).meld();
    switch (meld.type()) {
      case CHI, PON, DAIMINKAN -> table.call(seat, meld.id());
      case ANKAN -> table.closedKan(seat, meld.base());
      case KAKAN -> table.addedKan(seat, meld.base());
    }
    if (seat == table.seat()) {
      return meld.type() == Meld.Type.CHI || meld.type() == Meld.Type.PON ? decide(0, true) : List.of();
    }
    int offers = TenhouCodec.offers(tag);
    return meld.type() == Meld.Type.KAKAN && offers != 0 ? decide(offers, false) : List.of();
  }

  private void riichi(TenhouTag tag) {
    LiveTable table = table();
    int seat = round.seat(tag.integer("who"));
    switch (TenhouRound.ReachStep.of(tag)) {
      case DECLARE -> table.declareRiichi(seat);
      case ACCEPT -> table.acceptRiichi(seat);
    }
  }

  /**
   * 自分の手を、サーバが出した手の中から Decider に選ばせ、返すメッセージを作る。応答で見送りのほかに出せる手が無ければ、
   * 見送りを返す。手番で選べる手が無ければ IllegalStateException を投げる。
   *
   * @param offers メッセージの t（サーバが出した手のビット）
   * @param turn 手番（ツモの後とチー・ポンの後）の判断なら true、他家への応答なら false
   * @return 返すメッセージ
   */
  private List<String> decide(int offers, boolean turn) {
    Decisions decisions = live.decisions();
    int engine = decisions.size() == 0 ? 0 : TenhouCodec.bits(decisions, 0, turn);
    int server = offers & TenhouCodec.Offer.bits(turn);
    if (engine != server) {
      log.warn("Tenhou offers bits {} to seat {}, but the engine allows {}", server, live.seat(), engine);
    }
    int code = live.decide(each -> TenhouCodec.offered(each, offers));
    if (code == LiveTable.NO_DECISION) {
      if (turn) {
        throw new IllegalStateException("Seat " + live.seat() + " has no action on its turn");
      }
      return List.of(TenhouCodec.pass());
    }
    live.tiles(code, tiles);
    return TenhouCodec.reply(Action.of(code), tiles);
  }

  /**
   * AGARI か RYUUKYOKU を受ける。終局なら最後の局を照らし合わせて終わる。そうでなければ局の終わりとして、NEXTREADY を 1 回
   * 返す。
   */
  private List<String> roundEnd(TenhouTag tag) {
    LiveTable table = table();
    if (result != Result.UNREADABLE) {
      try {
        if (tag.name().equals("AGARI")) {
          round.agari(tag);
        } else {
          round.ryuukyoku(tag);
        }
        result = Result.READ;
      } catch (IllegalArgumentException failure) {
        unreadable(failure);
      }
    }
    if (tag.get("owari") == null) {
      return ready();
    }
    ended = true;
    if (result == Result.READ) {
      RoundResult last = result();
      if (last != null) {
        table.checkGame(last, round.finalScores());
      }
    }
    log.info("Tenhou game ends with {} mismatched rounds", mismatches());
    return List.of();
  }

  /** 読んだ局の結果から RoundResult を作る。作れなければ、読めなかった局に数えて null を返す。 */
  private RoundResult result() {
    try {
      return round.result(live.riichiMask());
    } catch (IllegalArgumentException failure) {
      unreadable(failure);
      return null;
    }
  }

  private void unreadable(IllegalArgumentException failure) {
    result = Result.UNREADABLE;
    unreadable++;
    log.warn("Tenhou round result cannot be read: {}", failure.getMessage());
  }

  private LiveTable table() {
    if (live == null) {
      throw new IllegalStateException("Tenhou sends a round message before INIT");
    }
    return live;
  }
}
