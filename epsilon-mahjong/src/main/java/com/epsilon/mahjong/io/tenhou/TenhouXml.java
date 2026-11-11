package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.io.tenhou.TenhouRound.DrawType;
import com.epsilon.mahjong.io.tenhou.TenhouRound.ReachStep;
import com.epsilon.mahjong.io.tenhou.TenhouRound.Seed;
import com.epsilon.mahjong.io.tenhou.TenhouRound.Ten;
import com.epsilon.mahjong.io.tenhou.TenhouRound.Yaku;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;

/**
 * 天鳳の牌譜の XML（mjloggm）を読み書きする。四人打ちの南喰赤と、三人打ちの三南喰赤を扱う。
 *
 * <p>読むときは、牌譜のイベントを RecordedRound の契約どおりの順に並べ直す。局の始まりと結果は TenhouRound で読む。北抜きは N タグに
 * 北抜きのコードとして書かれ、続く補充のツモは嶺上と同じくツモのタグで書かれる。抜きドラの翻は、和了の役のドラの翻に含まれる。
 * 打ち手の名前は、UN の n0〜n3 を URL エンコードとして復号したもの。牌譜の途中の UN（切断から戻ったとき）は、書いてある席の名前だけを
 * 替える。
 *
 * <p>書くときは、SHUFFLE・段位・レート・サイコロを書かない。名前は UTF-8 の各バイトを %XX にして UN に書く。鳴きで手から出す物理牌は、
 * 手の中の同じ種類の牌から、何枚目かの小さい順に選ぶ（手の赤を使う面子では赤を先に選ぶ）。大明槓・加槓の後めくりの表示牌は、天鳳の
 * 並びのとおり打牌の前に書く。和了の m には、天鳳と同じく面子と北抜きのコードを新しい順に並べる（加槓は元のポンの位置に置く）。
 */
public final class TenhouXml {

  private static final String VERSION = "2.3";

  /** UN の名前のバイトを書く 16 進の書式（大文字）。 */
  private static final HexFormat HEX = HexFormat.of().withUpperCase();

  /** owari の順位の点は千点単位で小数 1 桁まで書くので、百点（千点の 10 分の 1）単位で計算する。 */
  private static final int TENTHS = 10;

  private static final String SEPARATOR = ",";

  /** AGARI・RYUUKYOKU の ba の値の並び（本場と、局の終わりに場にある供託の本数）。 */
  private enum Ba {
    HONBA,
    KYOTAKU
  }

  private TenhouXml() {}

  /**
   * 天鳳の牌譜の XML を読む。読めない牌譜なら IllegalArgumentException を投げる。
   *
   * @param xml 牌譜の XML
   * @param id 牌譜 id。読んだ半荘のメタ情報に入れ、エラーメッセージの先頭にも付ける
   */
  public static RecordedGame read(String xml, String id) {
    return new Reader(id).read(xml);
  }

  /**
   * 半荘を天鳳の牌譜の XML に書く。局の結果の内訳を持たない局があれば（mjai から読んだ記録など）、
   * IllegalArgumentException を投げる。
   */
  public static String write(RecordedGame game) {
    return new Writer(game.variant()).write(game);
  }

  /**
   * GO の type から卓のルールを決める。三人打ちの bit があれば三人打ち、無ければ四人打ち。
   *
   * @throws IllegalArgumentException 赤なし・喰いなし・東風の卓のとき
   */
  public static Variant variant(int type) {
    Variant variant = TenhouRule.THREE_PLAYER.in(type) ? Variant.TENHOU_3P : Variant.TENHOU_4P;
    if (TenhouRule.NO_RED.in(type) || TenhouRule.NO_OPEN_TANYAO.in(type) || !TenhouRule.EAST_SOUTH.in(type)) {
      throw new IllegalArgumentException("Tenhou type " + type + " is not played by " + variant);
    }
    return variant;
  }

  private static String join(int[] values) {
    StringJoiner joined = new StringJoiner(SEPARATOR);
    for (int value : values) {
      joined.add(Integer.toString(value));
    }
    return joined.toString();
  }

  /** UN に書く名前を作る。UTF-8 の各バイトを %XX（16 進の大文字）にする。 */
  private static String encodeName(String name) {
    StringBuilder encoded = new StringBuilder();
    for (byte value : name.getBytes(StandardCharsets.UTF_8)) {
      encoded.append('%').append(HEX.toHexDigits(value));
    }
    return encoded.toString();
  }

  /** 1 つの牌譜を読む。局の状態は INIT ごとに作り直す。 */
  private static final class Reader {

    private final String id;
    private final List<RecordedRound> rounds = new ArrayList<>();
    private final List<RecordEvent> events = new ArrayList<>();

    /** タグの席ごとの打ち手の名前。UN に無ければ空の文字列。 */
    private final String[] names = new String[TenhouRound.TAG_SEATS];

    private Variant variant;
    private TileSet set;
    private int seats;
    private TenhouRound round;

    private RoundStart start;
    private int[][] deal;

    /** 席ごとに、最後にツモった牌。打牌か鳴きをするまで覚える。 */
    private int[] drawn;

    private int lastDiscarder;
    private int lastDiscard;

    /** 次の打牌が立直の宣言牌になる席（席ごとのビット）。 */
    private int declared;

    /** この局で立直が成立した席（席ごとのビット）。 */
    private int accepted;

    Reader(String id) {
      this.id = Objects.requireNonNull(id, "id");
      Arrays.fill(names, "");
    }

    RecordedGame read(String xml) {
      try {
        for (TenhouTag tag : TenhouTag.parse(xml)) {
          accept(tag);
        }
        finishRound();
        if (rounds.isEmpty()) {
          throw new IllegalArgumentException("The log has no round");
        }
        int[] finalScores = round.finalScores();
        if (finalScores == null) {
          throw new IllegalArgumentException("The log ends before the game does");
        }
        RecordedGame.Meta meta =
            new RecordedGame.Meta(id, RecordedGame.Source.TENHOU, List.of(Arrays.copyOf(names, seats)));
        return new RecordedGame(variant, rounds, finalScores, meta);
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException(id + ": " + failure.getMessage(), failure);
      }
    }

    private void accept(TenhouTag tag) {
      switch (tag.name()) {
        case "mjloggm", "SHUFFLE", "TAIKYOKU", "BYE", "PROF" -> {}
        case "UN" -> {
          for (int seat = 0; seat < names.length; seat++) {
            String name = tag.get("n" + seat);
            if (name != null) {
              names[seat] = URLDecoder.decode(name, StandardCharsets.UTF_8);
            }
          }
        }
        case "GO" -> rules(tag);
        case "INIT" -> init(tag);
        case "N" -> meld(tag);
        case "REACH" -> riichi(tag);
        case "DORA" -> {
          requireRound(tag);
          events.add(new RecordEvent.DoraReveal(TenhouTiles.tile(tag.integer("hai"), set)));
        }
        case "AGARI" -> {
          requireRound(tag);
          round.agari(tag);
        }
        case "RYUUKYOKU" -> {
          requireRound(tag);
          round.ryuukyoku(tag);
        }
        default -> move(tag);
      }
    }

    private void rules(TenhouTag tag) {
      variant = variant(tag.integer("type"));
      set = variant.tiles();
      seats = variant.seats();
      drawn = new int[seats];
      round = new TenhouRound(variant);
    }

    private void init(TenhouTag tag) {
      if (variant == null) {
        throw new IllegalArgumentException("A round starts before GO");
      }
      finishRound();
      start = round.start(tag);
      if (round.rotation() != 0) {
        throw new IllegalArgumentException("INIT names dealer " + tag.get("oya") + " in round " + start.round());
      }
      deal = new int[seats][];
      for (int seat = 0; seat < seats; seat++) {
        deal[seat] = TenhouTiles.tiles(tag.required("hai" + seat), set);
      }
      for (int empty = seats; empty < TenhouRound.TAG_SEATS; empty++) {
        String hand = tag.get("hai" + empty);
        if (hand != null && !hand.isEmpty()) {
          throw new IllegalArgumentException("INIT deals to the empty seat " + empty);
        }
      }
      events.clear();
      Arrays.fill(drawn, TurnEvent.NONE);
      lastDiscarder = TurnEvent.NONE;
      lastDiscard = TurnEvent.NONE;
      declared = 0;
      accepted = 0;
    }

    private void move(TenhouTag tag) {
      TenhouTiles.TileTag tile = variant == null ? null : TenhouTiles.TileTag.parse(tag.name(), variant);
      if (tile == null) {
        throw new IllegalArgumentException("Unknown Tenhou element " + tag.name());
      }
      requireRound(tag);
      int seat = tile.seat();
      switch (tile.move()) {
        case DRAW -> {
          drawn[seat] = tile.tile();
          events.add(new RecordEvent.Draw(seat, tile.tile()));
        }
        case DISCARD -> {
          events.add(
              new RecordEvent.Discard(seat, tile.tile(), tile.tile() == drawn[seat], (declared & 1 << seat) != 0));
          declared &= ~(1 << seat);
          drawn[seat] = TurnEvent.NONE;
          lastDiscarder = seat;
          lastDiscard = tile.tile();
        }
      }
    }

    private void meld(TenhouTag tag) {
      requireRound(tag);
      int tagSeat = tag.integer("who");
      int who = round.seat(tagSeat);
      int m = tag.integer("m");
      drawn[who] = TurnEvent.NONE;
      if (TenhouMeldCode.nuki(m)) {
        events.add(new RecordEvent.Nuki(who, TenhouMeldCode.nukiTile(m, variant)));
        return;
      }
      TenhouMeldCode code = TenhouMeldCode.decode(m, variant, tagSeat);
      Meld meld = code.meld();
      RecordEvent event =
          switch (meld.type()) {
            case CHI, PON, DAIMINKAN -> {
              if (code.called() != lastDiscard || meld.source().seat(who, seats) != lastDiscarder) {
                throw new IllegalArgumentException(
                    "Seat " + who + " makes " + meld + " on tile " + code.called() + ", but the last discard is "
                        + lastDiscard + " of seat " + lastDiscarder);
              }
              yield new RecordEvent.Call(who, meld.id());
            }
            case ANKAN -> new RecordEvent.ClosedKan(who, meld.base());
            case KAKAN -> new RecordEvent.AddedKan(who, meld.base());
          };
      events.add(event);
    }

    private void riichi(TenhouTag tag) {
      requireRound(tag);
      int who = round.seat(tag.integer("who"));
      switch (ReachStep.of(tag)) {
        case DECLARE -> declared |= 1 << who;
        case ACCEPT -> {
          accepted |= 1 << who;
          events.add(new RecordEvent.RiichiAccepted(who));
        }
      }
    }

    private void finishRound() {
      if (start == null) {
        return;
      }
      RoundResult result = round.result(accepted);
      rounds.add(new RecordedRound(start, deal, round.indicator(), RecordedRound.contractOrder(events), result));
      start = null;
    }

    private void requireRound(TenhouTag tag) {
      if (start == null) {
        throw new IllegalArgumentException(tag.name() + " comes outside a round");
      }
    }
  }

  /** 1 つの牌譜を書く。手と面子は局ごとに配牌から追う。 */
  private static final class Writer {

    private final Variant variant;
    private final TileSet set;
    private final int seats;
    private final StringBuilder xml = new StringBuilder();

    /** 席ごと・物理 id ごとに、その牌が手にあるか。 */
    private final boolean[][] held;

    /** 席ごとに、N タグに書いた面子と北抜きのコード（書いた順）。加槓は元のポンのコードと置き換える。 */
    private final List<List<Integer>> exposed = new ArrayList<>();

    /** 席ごとに、最後にツモった牌。 */
    private final int[] drawn;

    /** 席ごとのいまの点。局の始まりの点から始め、立直の供託を払えば引き、AGARI を 1 つ書くたびにその点の移動を足す。 */
    private final int[] scores;

    /** 順位ごとのウマ（千点）。 */
    private final int[] uma;

    /** めくったドラ表示牌（最初の表示牌から順に）。 */
    private final int[] indicators = new int[WinContext.MAX_INDICATORS];

    private int indicatorCount;

    /** チー・ポン・大明槓をした席（席ごとのビット）。 */
    private int openMask;

    /** ロンされうる最後の牌（打牌か、加槓で足した牌か、抜いた北）。 */
    private int claimable;

    Writer(Variant variant) {
      this.variant = variant;
      set = variant.tiles();
      seats = variant.seats();
      held = new boolean[seats][Tiles.PHYSICAL_IDS];
      drawn = new int[seats];
      scores = new int[seats];
      uma =
          switch (variant) {
            case TENHOU_4P -> new int[] {20, 10, -10, -20};
            case TENHOU_3P -> new int[] {20, 0, -20};
          };
      for (int seat = 0; seat < seats; seat++) {
        exposed.add(new ArrayList<>());
      }
    }

    String write(RecordedGame game) {
      xml.append("<mjloggm ver=\"").append(VERSION).append("\">");
      begin("GO");
      attribute("type", TenhouRule.hanchan(variant));
      attribute("lobby", 0);
      end();
      begin("UN");
      for (int seat = 0; seat < TenhouRound.TAG_SEATS; seat++) {
        attribute("n" + seat, seat < seats ? encodeName(game.meta().names().get(seat)) : "");
      }
      end();
      begin("TAIKYOKU");
      attribute("oya", 0);
      end();
      List<RecordedRound> rounds = game.rounds();
      for (int index = 0; index < rounds.size(); index++) {
        round(rounds.get(index), index == rounds.size() - 1 ? game.finalScores() : null);
      }
      return xml.append("</mjloggm>").toString();
    }

    /**
     * 1 局を書く。記録が局の結果の内訳を持たなければ IllegalArgumentException を投げる。
     *
     * @param finalScores 最後の局なら終局の点、ほかは null
     */
    private void round(RecordedRound round, int[] finalScores) {
      if (!round.result().detailed()) {
        throw new IllegalArgumentException("A Tenhou log needs the details of " + round.result());
      }
      RoundStart start = round.start();
      for (int seat = 0; seat < seats; seat++) {
        Arrays.fill(held[seat], false);
        exposed.get(seat).clear();
        scores[seat] = start.score(seat);
      }
      Arrays.fill(drawn, TurnEvent.NONE);
      indicatorCount = 0;
      openMask = 0;
      claimable = TurnEvent.NONE;
      int[] seed = new int[Seed.values().length];
      seed[Seed.ROUND.ordinal()] = TenhouRoundIndex.tenhou(start.round(), variant);
      seed[Seed.HONBA.ordinal()] = start.honba();
      seed[Seed.KYOTAKU.ordinal()] = start.kyotaku();
      seed[Seed.INDICATOR.ordinal()] = round.indicator();
      begin("INIT");
      attribute("seed", join(seed));
      attribute("ten", hundreds(scores));
      attribute("oya", variant.dealer(start.round()));
      int[][] deal = round.deal();
      for (int seat = 0; seat < TenhouRound.TAG_SEATS; seat++) {
        attribute("hai" + seat, seat < seats ? join(deal[seat]) : "");
      }
      for (int seat = 0; seat < seats; seat++) {
        for (int tile : deal[seat]) {
          held[seat][tile] = true;
        }
      }
      end();
      indicators[indicatorCount++] = round.indicator();
      List<RecordEvent> events = round.events();
      for (int index = 0; index < events.size(); index++) {
        switch (events.get(index)) {
          case RecordEvent.Draw draw -> {
            tile(TenhouTiles.Move.DRAW, draw.seat(), draw.tile());
            held[draw.seat()][draw.tile()] = true;
            drawn[draw.seat()] = draw.tile();
          }
          case RecordEvent.Discard discard -> index = discard(events, index, discard);
          case RecordEvent.Call call -> call(call);
          case RecordEvent.ClosedKan kan -> closedKan(kan);
          case RecordEvent.AddedKan kan -> addedKan(kan);
          case RecordEvent.DoraReveal reveal -> dora(reveal.tile());
          case RecordEvent.RiichiAccepted riichi -> {
            scores[riichi.seat()] -= Variant.DEPOSIT;
            begin("REACH");
            attribute("who", riichi.seat());
            attribute("ten", hundreds(scores));
            attribute("step", ReachStep.ACCEPT.value());
            end();
          }
          case RecordEvent.Nuki nuki -> nuki(nuki);
        }
      }
      if (round.result().kind().win()) {
        wins(round, finalScores);
      } else {
        drawResult(round, finalScores);
      }
    }

    /**
     * 打牌を書く。書く順は、直後にめくる表示牌、REACH の step 1（立直の宣言牌のときだけ）、打牌。ここで書いた最後のイベント（打牌か、
     * 後に続く表示牌）が、イベントの列の何番目かを返す。
     */
    private int discard(List<RecordEvent> events, int index, RecordEvent.Discard discard) {
      int last = index;
      while (last + 1 < events.size() && events.get(last + 1) instanceof RecordEvent.DoraReveal reveal) {
        dora(reveal.tile());
        last++;
      }
      if (discard.riichi()) {
        begin("REACH");
        attribute("who", discard.seat());
        attribute("step", ReachStep.DECLARE.value());
        end();
      }
      tile(TenhouTiles.Move.DISCARD, discard.seat(), discard.tile());
      held[discard.seat()][discard.tile()] = false;
      drawn[discard.seat()] = TurnEvent.NONE;
      claimable = discard.tile();
      return last;
    }

    private void call(RecordEvent.Call call) {
      int seat = call.seat();
      Meld meld = Meld.of(call.meld());
      int[] hand = new int[meld.size() - 1];
      boolean red = meld.aka() == Meld.AkaSource.CONSUMED_HAND_TILE;
      boolean calledSkipped = false;
      int size = 0;
      for (int index = 0; index < meld.size(); index++) {
        int kind = meld.tile(index);
        if (!calledSkipped && kind == meld.called()) {
          calledSkipped = true;
          continue;
        }
        int tile = take(seat, kind, red);
        red &= !set.red(tile);
        hand[size++] = tile;
      }
      meld(seat, new TenhouMeldCode(meld, claimable, hand));
      openMask |= 1 << seat;
    }

    private void closedKan(RecordEvent.ClosedKan kan) {
      int[] hand = new int[Tiles.COPIES];
      boolean red = false;
      for (int copy = 0; copy < Tiles.COPIES; copy++) {
        hand[copy] = take(kan.seat(), kan.kind(), true);
        red |= set.red(hand[copy]);
      }
      meld(kan.seat(), new TenhouMeldCode(Meld.ankan(kan.kind(), red), -1, hand));
    }

    private void addedKan(RecordEvent.AddedKan kan) {
      int seat = kan.seat();
      List<Integer> codes = exposed.get(seat);
      for (int index = 0; index < codes.size(); index++) {
        if (TenhouMeldCode.nuki(codes.get(index))) {
          continue;
        }
        TenhouMeldCode pon = TenhouMeldCode.decode(codes.get(index), variant, seat);
        if (pon.meld().type() == Meld.Type.PON && pon.meld().base() == kan.kind()) {
          int added = take(seat, kan.kind(), true);
          Meld meld = Meld.kakan(pon.meld(), set.red(added));
          int code = new TenhouMeldCode(meld, pon.called(), new int[] {added}).encode(variant, seat);
          n(seat, code);
          codes.set(index, code);
          claimable = added;
          return;
        }
      }
      throw new IllegalArgumentException("Seat " + seat + " has no pon of kind " + kan.kind() + " to add to");
    }

    private void nuki(RecordEvent.Nuki nuki) {
      if (!held[nuki.seat()][nuki.tile()]) {
        throw new IllegalArgumentException("Seat " + nuki.seat() + " does not hold north " + nuki.tile());
      }
      held[nuki.seat()][nuki.tile()] = false;
      int code = TenhouMeldCode.encodeNuki(nuki.tile());
      n(nuki.seat(), code);
      exposed.get(nuki.seat()).add(code);
      claimable = nuki.tile();
    }

    private void meld(int seat, TenhouMeldCode meld) {
      int code = meld.encode(variant, seat);
      n(seat, code);
      exposed.get(seat).add(code);
    }

    private void n(int seat, int code) {
      begin("N");
      attribute("who", seat);
      attribute("m", code);
      end();
    }

    /**
     * 手から、その種類の牌を 1 枚出す。赤を先に出すときは赤を、そうでないときは赤でない牌を優先し、何枚目かの小さい順に選ぶ。
     * 優先する牌が無ければ、残りの牌を出す。その種類の牌が手に無ければ IllegalArgumentException を投げる。
     *
     * @param red 赤を先に出すか
     */
    private int take(int seat, int kind, boolean red) {
      int fallback = TurnEvent.NONE;
      for (int copy = 0; copy < Tiles.COPIES; copy++) {
        int tile = Tiles.physicalId(kind, copy);
        if (!held[seat][tile]) {
          continue;
        }
        if (set.red(tile) == red) {
          held[seat][tile] = false;
          return tile;
        }
        if (fallback == TurnEvent.NONE) {
          fallback = tile;
        }
      }
      if (fallback == TurnEvent.NONE) {
        throw new IllegalArgumentException("Seat " + seat + " holds no tile of kind " + kind);
      }
      held[seat][fallback] = false;
      return fallback;
    }

    private void dora(int tile) {
      begin("DORA");
      attribute("hai", tile);
      end();
      indicators[indicatorCount++] = tile;
    }

    private void tile(TenhouTiles.Move move, int seat, int tile) {
      xml.append('<').append(new TenhouTiles.TileTag(move, seat, tile).name()).append("/>");
    }

    /**
     * AGARI を書く（ダブロンは和了者ごとに 1 つ）。2 人目からの和了者の点の移動は、本場を付けないロンの精算として Settlement で
     * 計算する。局の点の移動からそれらを引いた残りが、最初の和了者の点の移動になる。ba の本場はどの AGARI にも書き、供託は最初の AGARI にだけ書く。
     */
    private void wins(RecordedRound round, int[] finalScores) {
      RoundResult result = round.result();
      RoundStart start = round.start();
      int dealer = variant.dealer(start.round());
      int riichiMask = round.riichiMask();
      boolean tsumo = result.kind() == EndKind.TSUMO;
      List<RoundResult.Win> wins = result.wins();
      int[][] parts = new int[wins.size()][seats];
      parts[0] = moved(round);
      for (int index = 1; index < wins.size(); index++) {
        RoundResult.Win win = wins.get(index);
        Settlement.ron(variant, win.base(), win.seat(), result.loser(), dealer, 0, win.pao(), parts[index]);
        for (int seat = 0; seat < seats; seat++) {
          parts[0][seat] -= parts[index][seat];
        }
      }
      for (int index = 0; index < wins.size(); index++) {
        RoundResult.Win win = wins.get(index);
        int winner = win.seat();
        int winning = tsumo ? drawn[winner] : claimable;
        boolean riichi = (riichiMask & 1 << winner) != 0;
        int[] ba = new int[Ba.values().length];
        ba[Ba.HONBA.ordinal()] = start.honba();
        if (index == 0) {
          ba[Ba.KYOTAKU.ordinal()] = deposits(round);
        }
        int[] ten = new int[Ten.values().length];
        ten[Ten.FU.ordinal()] = win.fu();
        ten[Ten.POINTS.ordinal()] = win.points(variant, dealer, tsumo);
        ten[Ten.LIMIT.ordinal()] = limit(win.base());
        int[] m = exposed.get(winner).reversed().stream().mapToInt(Integer::intValue).toArray();
        begin("AGARI");
        attribute("ba", join(ba));
        attribute("hai", join(hand(winner, tsumo ? TurnEvent.NONE : winning)));
        if (m.length > 0) {
          attribute("m", join(m));
        }
        attribute("machi", winning);
        attribute("ten", join(ten));
        if ((win.yaku() & ScoringYaku.YAKUMAN_BITS) != 0) {
          attribute("yakuman", yakuman(win, start.round()));
        } else {
          attribute("yaku", yaku(win, start.round(), riichi));
        }
        attribute("doraHai", join(Arrays.copyOf(indicators, indicatorCount)));
        if (riichi && result.uraCount() > 0) {
          attribute("doraHaiUra", join(result.uraIndicators()));
        }
        attribute("who", winner);
        attribute("fromWho", tsumo ? winner : result.loser());
        if (win.pao() != RoundOutcome.NO_SEAT) {
          attribute("paoWho", win.pao());
        }
        attribute("sc", pairs(parts[index]));
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] += parts[index][seat];
        }
        if (finalScores != null && index == wins.size() - 1) {
          attribute("owari", owari(finalScores));
        }
        end();
      }
    }

    private void drawResult(RecordedRound round, int[] finalScores) {
      RoundResult result = round.result();
      begin("RYUUKYOKU");
      if (result.kind() == EndKind.ABORTIVE) {
        attribute("type", DrawType.of(result.abortive()).text());
      } else if (result.nagashiMask() != 0) {
        attribute("type", DrawType.NM.text());
      }
      int[] ba = new int[Ba.values().length];
      ba[Ba.HONBA.ordinal()] = round.start().honba();
      ba[Ba.KYOTAKU.ordinal()] = deposits(round);
      attribute("ba", join(ba));
      attribute("sc", pairs(moved(round)));
      for (int seat = 0; seat < seats; seat++) {
        if ((result.tenpaiMask() & 1 << seat) != 0) {
          attribute("hai" + seat, join(hand(seat, TurnEvent.NONE)));
        }
      }
      if (finalScores != null) {
        attribute("owari", owari(finalScores));
      }
      end();
    }

    /** sc に書く点の移動を返す。記録の点の移動に、この局の立直の供託を戻したもの。 */
    private int[] moved(RecordedRound round) {
      int[] moved = round.result().deltas();
      for (int seat = 0; seat < seats; seat++) {
        if ((round.riichiMask() & 1 << seat) != 0) {
          moved[seat] += Variant.DEPOSIT;
        }
      }
      return moved;
    }

    /** 局の終わりに場にある供託の本数。 */
    private int deposits(RecordedRound round) {
      return round.start().kyotaku() + Integer.bitCount(round.riichiMask());
    }

    /**
     * 席の手牌を、物理 id の昇順で返す。
     *
     * @param extra 足す牌。足さなければ {@code TurnEvent.NONE}
     */
    private int[] hand(int seat, int extra) {
      int[] tiles = new int[Tiles.PHYSICAL_IDS];
      int size = 0;
      for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
        if (held[seat][tile] || tile == extra) {
          tiles[size++] = tile;
        }
      }
      return Arrays.copyOf(tiles, size);
    }

    private String yakuman(RoundResult.Win win, int round) {
      StringJoiner numbers = new StringJoiner(SEPARATOR);
      for (ScoringYaku yaku : ScoringYaku.values()) {
        if ((win.yaku() & yaku.bit()) != 0) {
          numbers.add(Integer.toString(Yaku.number(yaku, variant, round, win.seat())));
        }
      }
      return numbers.toString();
    }

    /**
     * 役と翻の組を役の番号の順に並べ、続けてドラ・赤・裏ドラを書く。ドラと赤は 1 翻以上のときだけ書き、裏ドラは立直した和了なら
     * 0 翻でも書く。翻の和が記録と違えば IllegalArgumentException を投げる。
     */
    private String yaku(RoundResult.Win win, int round, boolean riichi) {
      boolean closed = (openMask & 1 << win.seat()) == 0;
      int[] han = new int[Yaku.VALUES.length];
      int total = 0;
      for (ScoringYaku yaku : ScoringYaku.values()) {
        if ((win.yaku() & yaku.bit()) != 0) {
          han[Yaku.number(yaku, variant, round, win.seat())] = yaku.han(closed);
          total += yaku.han(closed);
        }
      }
      StringJoiner pairs = new StringJoiner(SEPARATOR);
      for (Yaku yaku : Yaku.VALUES) {
        if (han[yaku.ordinal()] > 0) {
          pairs.add(yaku.ordinal() + SEPARATOR + han[yaku.ordinal()]);
        }
      }
      if (win.dora() > 0) {
        pairs.add(Yaku.DORA.ordinal() + SEPARATOR + win.dora());
      }
      if (win.aka() > 0) {
        pairs.add(Yaku.AKA.ordinal() + SEPARATOR + win.aka());
      }
      if (riichi) {
        pairs.add(Yaku.URA.ordinal() + SEPARATOR + win.ura());
      }
      if (total + win.dora() + win.aka() + win.ura() != win.han()) {
        throw new IllegalArgumentException("The yaku of " + win + " do not add up to its han");
      }
      return pairs.toString();
    }

    /** ten の LIMIT を返す。満貫未満 0、満貫 1 … 三倍満 4、役満は倍率に依らず 5。 */
    private static int limit(int base) {
      int over = BaseClass.of(base).ordinal() - BaseClass.MANGAN.ordinal();
      return over < 0 ? 0 : 1 + Math.min(over, BaseClass.YAKUMAN.ordinal() - BaseClass.MANGAN.ordinal());
    }

    /**
     * owari の値を作る。席ごとに、終局の点（百点単位）と順位の点（(点 − 返し点) ／ 1000 ＋ ウマ、トップはオカも足す）を並べる。
     * 空き席は、終局の点も順位の点も 0。
     */
    private String owari(int[] finalScores) {
      int[] ranks = new int[seats];
      Placement.ranks(finalScores, ranks);
      int oka = (variant.returnPoints() - variant.startPoints()) * seats;
      StringJoiner values = new StringJoiner(SEPARATOR);
      for (int seat = 0; seat < TenhouRound.TAG_SEATS; seat++) {
        int score = 0;
        int tenths = 0;
        if (seat < seats) {
          score = finalScores[seat];
          int points = score - variant.returnPoints() + (ranks[seat] == 0 ? oka : 0);
          tenths = points / TenhouRound.POINT_UNIT + uma[ranks[seat]] * TENTHS;
        }
        values.add(Integer.toString(score / TenhouRound.POINT_UNIT));
        values.add((tenths < 0 ? "-" : "") + Math.abs(tenths) / TENTHS + "." + Math.abs(tenths) % TENTHS);
      }
      return values.toString();
    }

    /** sc の値を作る。席ごとに、いまの点と点の移動（百点単位）を並べる。空き席はどちらも 0。 */
    private String pairs(int[] change) {
      StringJoiner values = new StringJoiner(SEPARATOR);
      for (int seat = 0; seat < TenhouRound.TAG_SEATS; seat++) {
        values.add(Integer.toString(seat < seats ? scores[seat] / TenhouRound.POINT_UNIT : 0));
        values.add(Integer.toString(seat < seats ? change[seat] / TenhouRound.POINT_UNIT : 0));
      }
      return values.toString();
    }

    /** 席ごとの点（百点単位）。空き席は 0。 */
    private static String hundreds(int[] points) {
      int[] hundreds = new int[TenhouRound.TAG_SEATS];
      for (int seat = 0; seat < points.length; seat++) {
        hundreds[seat] = points[seat] / TenhouRound.POINT_UNIT;
      }
      return join(hundreds);
    }

    private void begin(String name) {
      xml.append('<').append(name);
    }

    private void attribute(String key, Object value) {
      xml.append(' ').append(key).append("=\"").append(value).append('"');
    }

    private void end() {
      xml.append("/>");
    }
  }
}
