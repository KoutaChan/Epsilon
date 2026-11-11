package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * 牌山の運だけを変えて局を打ち直すテスト用のヘルパー（ほかのモジュールのテストも使う）。牌譜の局を記録した手で打ち、判断の時点で、まだ
 * 読まれていない牌山の位置だけを並べ替えて、同じ手で打ち直す。打ち直したエンジンは、全ての席の手牌、公開のイベント、偶然のイベント、判断が
 * 元のエンジンと同じことを確かめてから渡す。並べ替えるのは誰にも見えない牌だけなので、渡したエンジンを先へ進めたときに出る牌は、その時点で
 * 誰にも見えない牌を並べ替えたものになる。
 *
 * <p>読まれた位置は、配牌（[0, 13S)、S は席の数）、山のツモ、めくった表示牌、嶺上・補充の位置で、どれも {@code Wall} の位置のメソッドで
 * 決まる。裏ドラ表示牌の位置は局末まで読まれないので、並べ替えの対象に入る。
 */
public final class WallReplay {

  private final RoundStart start;
  private final Variant variant;
  private final int[] wall;
  private final RuleWorkspace ws;
  private final Engine engine;
  private final List<int[]> moves = new ArrayList<>();
  private Boundary boundary;

  /**
   * 牌譜の局を始め、最初の判断の時点まで進める。牌山の長さがルールの牌の数と違うか、牌の分からない位置があれば
   * IllegalArgumentException を投げる。
   *
   * @param start 局の始まりの卓
   * @param wall 位置ごとの物理 id を並べた牌山（コピーして持つ）
   */
  public WallReplay(RoundStart start, int[] wall) {
    this.start = Objects.requireNonNull(start, "start");
    variant = start.variant();
    if (wall.length != variant.tiles().size()) {
      throw new IllegalArgumentException(variant + " needs " + variant.tiles().size() + " positions, got " + wall.length);
    }
    for (int position = 0; position < wall.length; position++) {
      if (wall[position] == Wall.UNKNOWN) {
        throw new IllegalArgumentException("Wall position " + position + " is not known");
      }
    }
    this.wall = wall.clone();
    ws = new RuleWorkspace(variant);
    engine = new Engine(variant);
    engine.startRecordedRound(start, this.wall);
    boundary = engine.advance(ws);
  }

  /** 記録した手で打つ元のエンジン。進めるときは、このクラスの {@code commit} だけを使う。 */
  public Engine engine() {
    return engine;
  }

  /** 元のエンジンが止まっている所（判断の時点か局末）。 */
  public Boundary boundary() {
    return boundary;
  }

  /**
   * 元のエンジンを進め、選んだ手を記録する。判断の時点でなければ IllegalStateException を投げる。
   *
   * @param choices 判断する席ごとに選んだ手（合法手の何番目か）。先頭から判断する席の数だけを読む
   * @return 進めた後に止まった所
   */
  public Boundary commit(int... choices) {
    requireDecisions();
    int[] move = Arrays.copyOf(choices, engine.decisions().size());
    boundary = engine.commit(move, ws);
    moves.add(move);
    return boundary;
  }

  /**
   * 今の判断の時点までに読まれた位置の牌はそのまま残し、ほかの位置の牌だけを並べ替えた牌山を返す。判断の時点でなければ
   * IllegalStateException を投げる。
   *
   * @param seed 並べ替えの乱数の種
   * @return 位置ごとの物理 id を並べた新しい牌山
   */
  public int[] reshuffled(long seed) {
    requireDecisions();
    boolean[] read = readPositions();
    int[] unread = new int[wall.length];
    int count = 0;
    for (int position = 0; position < wall.length; position++) {
      if (!read[position]) {
        unread[count++] = position;
      }
    }
    int[] shuffled = wall.clone();
    Random random = new Random(seed);
    for (int index = count - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int tile = shuffled[unread[index]];
      shuffled[unread[index]] = shuffled[unread[other]];
      shuffled[unread[other]] = tile;
    }
    return shuffled;
  }

  /**
   * 並べ替えた牌山で局を始め、記録した手で今の判断の時点まで打ち直したエンジンを返す。判断の時点でなければ IllegalStateException を
   * 投げる。打ち直したエンジンのビューか判断が元のエンジンと違えば AssertionError を投げる。
   *
   * @param seed 並べ替えの乱数の種
   * @return 今の判断の時点で止まっている新しいエンジン
   */
  public Engine replay(long seed) {
    int[] shuffled = reshuffled(seed);
    Engine replayed = new Engine(variant);
    replayed.startRecordedRound(start, shuffled);
    Boundary reached = replayed.advance(ws);
    for (int[] move : moves) {
      if (reached != Boundary.DECISIONS) {
        throw new AssertionError("The replay left the recorded line at " + reached);
      }
      reached = replayed.commit(move, ws);
    }
    if (reached != Boundary.DECISIONS) {
      throw new AssertionError("The replay stopped at " + reached);
    }
    List<String> expected = describe(engine);
    List<String> actual = describe(replayed);
    for (int line = 0; line < Math.max(expected.size(), actual.size()); line++) {
      String recorded = line < expected.size() ? expected.get(line) : "(none)";
      String again = line < actual.size() ? actual.get(line) : "(none)";
      if (!recorded.equals(again)) {
        throw new AssertionError("The replay differs from the recorded line:\n  " + recorded + "\n  " + again);
      }
    }
    return replayed;
  }

  private void requireDecisions() {
    if (boundary != Boundary.DECISIONS) {
      throw new IllegalStateException("The recorded line is at " + boundary + ", not at a decision");
    }
  }

  /**
   * 元のエンジンが今までに読んだ位置を返す。読んだのは配牌、山のツモ、嶺上と補充、表示牌の位置で、どの位置の牌も、偶然のイベントや
   * ビューに出た牌と同じことを確かめる。
   */
  private boolean[] readPositions() {
    boolean[] read = new boolean[wall.length];
    for (int offset = 0; offset < variant.seats(); offset++) {
      for (int index = 0; index < Variant.HAND; index++) {
        read[Wall.dealtPosition(variant, offset, index)] = true;
      }
    }
    FullView full = engine.full();
    ChanceLog chance = full.chance();
    int draws = 0;
    int rinshans = 0;
    for (int index = 0; index < chance.size(); index++) {
      switch (chance.kind(index)) {
        case DRAW -> markRead(read, Wall.livePosition(variant, draws++), chance.tile(index));
        case RINSHAN -> markRead(read, Wall.rinshanPosition(variant, ++rinshans), chance.tile(index));
        case INDICATOR, URA -> {}
      }
    }
    SeatView table = full.seat(0);
    for (int index = 0; index < table.indicatorCount(); index++) {
      markRead(read, Wall.indicatorPosition(variant, index), table.indicator(index));
    }
    return read;
  }

  private void markRead(boolean[] read, int position, int tile) {
    if (wall[position] != tile) {
      throw new AssertionError(
          "Wall position " + position + " holds " + wall[position] + ", but " + tile + " was read");
    }
    read[position] = true;
  }

  /** エンジンのビュー（FullView と席ごとのビュー）と判断から読める全てを、文字列の行にして返す。 */
  private static List<String> describe(Engine engine) {
    FullView full = engine.full();
    Variant variant = full.variant();
    List<String> lines = new ArrayList<>();
    ChanceLog chance = full.chance();
    StringBuilder text = new StringBuilder("chance");
    for (int index = 0; index < chance.size(); index++) {
      text.append(' ').append(chance.kind(index)).append(chance.seat(index)).append('/').append(chance.tile(index));
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      text.append(' ').append(chance.start(kind)).append('/').append(chance.unseen(kind)).append('/')
          .append(full.unseen(kind));
    }
    lines.add(text.toString());
    SeatView table = full.seat(0);
    text = new StringBuilder("table round=").append(table.round()).append(" dealer=").append(table.dealer())
        .append(" current=").append(table.current()).append(" honba=").append(table.honba())
        .append(" kyotaku=").append(table.kyotaku()).append(" live=").append(table.liveLeft());
    EventLog events = table.events();
    for (int event = 0; event < events.size(); event++) {
      text.append(' ').append(events.kind(event)).append(events.seat(event)).append('/').append(events.tile(event))
          .append(events.red(event) ? "r" : "").append(events.tsumogiri(event) ? "t" : "")
          .append(events.riichi(event) ? "R" : "").append('/').append(events.calledBy(event))
          .append('/').append(events.meld(event));
    }
    for (int index = 0; index < table.indicatorCount(); index++) {
      text.append(" dora ").append(table.indicator(index)).append('@').append(table.indicatorEvent(index));
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      text.append(' ').append(table.visible(kind)).append('/').append(table.doraWeight(kind));
    }
    lines.add(text.toString());
    HandShape shape = new HandShape();
    for (int seat = 0; seat < variant.seats(); seat++) {
      SeatView view = full.seat(seat);
      full.copyHand(seat, shape);
      text = new StringBuilder("seat ").append(seat).append(' ').append(shape).append(" score=")
          .append(view.score(seat)).append(' ').append(view.riichi(seat)).append(view.ippatsu(seat) ? " ippatsu" : "")
          .append(" river=").append(Long.toHexString(view.riverKinds(seat)))
          .append(" safe=").append(Long.toHexString(view.riichiSafe(seat)))
          .append(" nuki=").append(view.nuki(seat)).append(" furiten=").append(full.furiten(seat))
          .append(" waits=").append(Long.toHexString(full.waits(seat))).append(" melds=");
      for (int index = 0; index < view.meldCount(seat); index++) {
        text.append(view.meld(seat, index)).append(';');
      }
      text.append(" held=");
      for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
        if (view.holds(tile)) {
          text.append(tile).append(',');
        }
      }
      TurnEvent last = view.last();
      text.append(" last=").append(last.kind()).append(last.seat()).append('/').append(last.tile())
          .append(last.red() ? "r" : "").append(last.rinshan() ? " rinshan" : "")
          .append(last.doraPending() ? " pending" : "").append(' ').append(last.kanKind());
      lines.add(text.toString());
    }
    Decisions decisions = engine.decisions();
    for (int slot = 0; slot < decisions.size(); slot++) {
      text = new StringBuilder("decision ").append(decisions.id(slot)).append(" seat ").append(decisions.seat(slot))
          .append(' ').append(decisions.kind(slot));
      for (int index = 0; index < decisions.legalCount(slot); index++) {
        text.append(' ').append(Action.of(decisions.legal(slot, index)));
      }
      lines.add(text.toString());
    }
    lines.add("decisions " + decisions.size());
    return lines;
  }
}
