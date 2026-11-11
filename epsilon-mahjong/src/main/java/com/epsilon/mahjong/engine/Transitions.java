package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;

/**
 * エンジンが 1 局を進める遷移。局を配り、選ばれた手を行って、次の判断の時点か局末まで進める。牌山・偶然のイベント・他家の応答・
 * 局末を持つ。公開のイベントと手は {@link Steps} で書く（1 席から見た卓と同じ手順）。エンジンが 1 つ持ち、局を進める呼び出しは
 * メモリを確保しない。
 *
 * <p>表示牌は牌山からめくる。暗槓ではすぐにめくり、明槓・加槓の後めくりは、次の打牌の後か続く嶺上のツモの直前にめくる。北抜きでは
 * 新しい表示牌をめくらない。保留の表示牌があれば、補充を引く前にめくる。嶺上・補充のツモで和了したときは、保留の表示牌はめくらない。
 * 嶺上・補充のツモで山の残りが 0 になったとき、そのツモの和了は嶺上開花で、海底は付かない。続く打牌は河底になる。
 *
 * <p>途中流局: 九種九牌は手番の席が選んだときに、三家和はロンを決めるときに起きる。四風連打・四家立直・四槓散了は、打牌が通った後に
 * 起きる。
 */
final class Transitions {

  /** 半荘・牌譜の局の最初の判断 ID。 */
  private static final long FIRST_ID = 1;

  private final int seats;
  private final Wall wall;
  private final Steps steps;
  private final PublicState state;
  private final Hand[] hands;
  private final Decisions decisions;
  private final Window window;
  private final ChanceLog chance;
  private final Resolution resolution;
  private final RoundEnd end;
  private final FullReader full;

  /** 次に使う判断 ID。局をまたいで続き、半荘か牌譜の局を始めると初めに戻る。 */
  private long nextId;

  Transitions(Variant variant) {
    seats = variant.seats();
    wall = new Wall(variant);
    steps = new Steps(variant, variant.seatMask());
    state = steps.state();
    hands = steps.hands();
    decisions = steps.decisions();
    window = steps.window();
    chance = new ChanceLog(variant);
    resolution = new Resolution(variant);
    end = new RoundEnd(variant);
    full = new FullReader(state, hands, decisions, chance);
  }

  Decisions decisions() {
    return decisions;
  }

  Resolution resolution() {
    return resolution;
  }

  RoundEnd end() {
    return end;
  }

  FullView full() {
    return full;
  }

  /** 半荘を始める。判断 ID を初めに戻す。次の局からは、乱数の種と {@code epoch} から牌山を作る。 */
  void start(long seed, int epoch, DrawOrder drawOrder) {
    wall.start(seed, epoch, drawOrder);
    nextId = FIRST_ID;
  }

  /**
   * 次の局を牌譜の牌山で配る。判断 ID は初めに戻す。
   *
   * @param recorded 位置 → 物理 id の牌山。見えない位置は {@link Wall#UNKNOWN}
   */
  void load(int[] recorded) {
    wall.load(recorded);
    nextId = FIRST_ID;
  }

  /**
   * 卓の局を配り、最初の表示牌をめくって、親が最初にツモった後の手番まで進める。
   *
   * @param table 局の始めの卓
   * @param shuffle 次の牌山を作るなら true。false なら {@link #load} した牌山で配る
   */
  Boundary deal(TableState table, boolean shuffle, RuleWorkspace ws) {
    steps.start(table);
    int dealer = state.dealer();
    if (shuffle) {
      wall.shuffle(dealer);
    }
    for (int offset = 0; offset < seats; offset++) {
      Hand hand = hands[(dealer + offset) % seats];
      for (int index = 0; index < Variant.HAND; index++) {
        hand.add(wall.dealt(offset, index));
      }
    }
    int indicator = wall.indicator(0);
    state.reveal(indicator);
    chance.begin(hands, indicator);
    return draw(dealer, ws);
  }

  /**
   * 判断する席ごとに選んだ手から実際に行う手を決めて行い、次の判断の時点か局末まで進める。
   *
   * @param choices 席ごとに選んだ手（合法手の何番目か。範囲は呼ぶ側が確かめる）
   * @param tile 手番の席が打つか抜く物理 id（{@link Engine#commit(int[], int, RuleWorkspace)}）。エンジンが選ぶなら
   *     {@link TurnEvent#NONE}
   */
  Boundary commit(int[] choices, int tile, RuleWorkspace ws) {
    EngineDecisionKind kind = decisions.kind(0);
    if (tile != TurnEvent.NONE && kind != EngineDecisionKind.TURN && kind != EngineDecisionKind.POST_CALL_DAHAI) {
      throw new IllegalArgumentException("A " + kind + " decision plays no tile of its own, not tile " + tile);
    }
    return switch (kind) {
      case TURN, POST_CALL_DAHAI -> act(choices[0], tile, ws);
      case RESPONSE -> afterDiscard(window.close(state, hands, decisions, choices, resolution), ws);
      case CHANKAN, NUKI_RESPONSE -> afterKanOrNuki(window.close(state, hands, decisions, choices, resolution), ws);
    };
  }

  /**
   * 牌山のまだ読んでいない牌を並べ直す（{@link Wall} の並べ直し）。判断の時点でだけ呼ぶ。和了を合法手に持つ立直の席は、和了の点を
   * 新しい牌山の裏ドラで数え直す。
   *
   * @param seed 並べ直しに使う乱数の種
   */
  void reshuffleUnseen(long seed, RuleWorkspace ws) {
    wall.reshuffleUnseen(seed, chance);
    for (int slot = 0; slot < decisions.size(); slot++) {
      int seat = decisions.seat(slot);
      if (decisions.hasWin(slot) && state.riichi(seat) != RiichiState.NONE) {
        rescoreWin(slot, seat, ws);
      }
    }
  }

  void copyFrom(Transitions other) {
    wall.copyFrom(other.wall);
    steps.copyFrom(other.steps);
    chance.copyFrom(other.chance);
    resolution.copyFrom(other.resolution);
    end.copyFrom(other.end);
    nextId = other.nextId;
  }

  /**
   * 手番か、チー・ポンの後の打牌で、判断する 1 席が選んだ手を行う。
   *
   * @param tile 打つか抜く物理 id。エンジンが選ぶなら {@link TurnEvent#NONE}
   */
  private Boundary act(int choice, int tile, RuleWorkspace ws) {
    int seat = decisions.seat(0);
    EngineDecisionKind kind = decisions.kind(0);
    Action action = Action.of(decisions.legal(0, choice));
    int played =
        switch (action.type()) {
          case DAHAI, RIICHI_DAHAI -> steps.tileOf(seat, action, tile);
          case NUKI -> steps.northOf(seat, tile);
          default -> {
            if (tile != TurnEvent.NONE) {
              throw new IllegalArgumentException(action + " takes no tile out of the hand, not tile " + tile);
            }
            yield TurnEvent.NONE;
          }
        };
    resolution.clear();
    resolution.add(
        decisions.id(0), true, decisions.legalCount(0) == 1 ? Resolution.Role.FORCED : Resolution.Role.CAUSAL);
    return switch (action.type()) {
      case TSUMO_AGARI -> tsumo(seat, ws);
      case KYUSHU_KYUHAI -> abortive(AbortiveDraw.NINE_TERMINALS, ws);
      case ANKAN -> ankan(seat, action.tile(), ws);
      case KAKAN -> kakan(seat, action.tile(), ws);
      case NUKI -> nuki(seat, played, ws);
      case DAHAI, RIICHI_DAHAI -> discard(seat, action, played, ws);
      case CHI, PON, DAIMINKAN, RON_AGARI, PASS -> throw new IllegalStateException(kind + " cannot take " + action);
    };
  }

  /**
   * 山からツモる。山が尽きていれば荒牌で局を終える。
   *
   * @param seat ツモる席
   */
  private Boundary draw(int seat, RuleWorkspace ws) {
    if (wall.liveLeft() == 0) {
      return exhaustive(ws);
    }
    int tile = wall.draw(seat);
    chance.appendDraw(seat, tile);
    return turn(seat, tile, false, ws);
  }

  /**
   * 槓の嶺上か、北抜きの補充をツモる。
   *
   * @param seat 槓か北抜きをした席
   */
  private Boundary rinshan(int seat, RuleWorkspace ws) {
    int tile = wall.drawRinshan(seat);
    chance.appendRinshan(seat, tile);
    return turn(seat, tile, true, ws);
  }

  /** ツモった牌を手に入れて手番の判断を作る。手番には必ず打牌があるので、判断は空にならない。 */
  private Boundary turn(int seat, int tile, boolean rinshan, RuleWorkspace ws) {
    steps.draw(seat, tile, rinshan);
    nextId = Legal.turn(state, hands[seat], wall, decisions, nextId, ws);
    return Boundary.DECISIONS;
  }

  /**
   * 手番の席が打つ。保留の表示牌は打牌の後、打牌への応答を求める前にめくる。
   *
   * @param action 打牌か立直の打牌
   * @param tile 打つ物理 id
   */
  private Boundary discard(int seat, Action action, int tile, RuleWorkspace ws) {
    steps.discard(seat, tile, action.selection().tsumogiri(), action.type() == Action.Type.RIICHI_DAHAI);
    revealPending();
    nextId = steps.respond(wall, nextId, ws);
    if (decisions.size() > 0) {
      return Boundary.DECISIONS;
    }
    steps.pass();
    return afterDiscard(Window.Outcome.PASS, ws);
  }

  /** 打牌への応答が締まった後。ロンも途中流局も無ければ、鳴いた席か打った席の次の席へ進む。 */
  private Boundary afterDiscard(Window.Outcome outcome, RuleWorkspace ws) {
    AbortiveDraw abort = steps.abort();
    return switch (outcome) {
      case RON -> ron(ws);
      case TRIPLE_RON -> abortive(AbortiveDraw.TRIPLE_RON, ws);
      case CALL -> call(window.caller(), Action.of(window.call()), ws);
      case PASS -> abort == null ? draw((state.last().seat() + 1) % seats, ws) : abortive(abort, ws);
    };
  }

  /** 直前の打牌へのチー・ポン・大明槓。チー・ポンは鳴いた後の打牌の判断へ、大明槓は嶺上のツモへ進む。 */
  private Boundary call(int caller, Action action, RuleWorkspace ws) {
    steps.call(caller, steps.meldOf(caller, action));
    if (action.type() == Action.Type.DAIMINKAN) {
      return rinshan(caller, ws);
    }
    nextId = Legal.postCall(hands[caller], caller, decisions, nextId);
    return Boundary.DECISIONS;
  }

  /** 暗槓。保留の表示牌と、この槓の表示牌の順にめくってから嶺上を引く。 */
  private Boundary ankan(int seat, int kind, RuleWorkspace ws) {
    steps.ankan(seat, kind);
    revealPending();
    revealIndicator();
    return rinshan(seat, ws);
  }

  /** 加槓。純手牌の 4 枚目を同じ種類のポンに足し、加槓への応答（槍槓）を求める。 */
  private Boundary kakan(int seat, int kind, RuleWorkspace ws) {
    steps.kakan(seat, kind);
    return respondToKanOrNuki(ws);
  }

  /**
   * 北抜き。抜いた北への応答を求める。
   *
   * @param tile 抜く北の物理 id
   */
  private Boundary nuki(int seat, int tile, RuleWorkspace ws) {
    steps.nuki(seat, tile);
    return respondToKanOrNuki(ws);
  }

  /** 加槓か北抜きへの応答を求める。応答する席が無ければ、誰も応じなかったとして先へ進める。 */
  private Boundary respondToKanOrNuki(RuleWorkspace ws) {
    nextId = steps.respond(wall, nextId, ws);
    if (decisions.size() > 0) {
      return Boundary.DECISIONS;
    }
    steps.pass();
    return afterKanOrNuki(Window.Outcome.PASS, ws);
  }

  /**
   * 加槓か北抜きへの応答が締まった後。ロンが無ければ全員の一発を消し、保留の表示牌があればめくる。加槓なら、この槓の表示牌を
   * 後めくりにする。最後に、宣言した席が嶺上・補充を引く。
   */
  private Boundary afterKanOrNuki(Window.Outcome outcome, RuleWorkspace ws) {
    return switch (outcome) {
      case RON -> ron(ws);
      case TRIPLE_RON -> abortive(AbortiveDraw.TRIPLE_RON, ws);
      case PASS -> {
        state.clearIppatsu();
        revealPending();
        if (state.last().kind() == TurnEvent.Kind.KAN) {
          state.deferDora();
        }
        yield rinshan(state.last().seat(), ws);
      }
      case CALL -> throw new IllegalStateException("No call answers an added kan or a nuki");
    };
  }

  private Boundary tsumo(int seat, RuleWorkspace ws) {
    WinScore score = decisions.win(0);
    int pao = WinLegality.pao(full.seat(seat), score, RoundOutcome.NO_SEAT);
    end.outcome().setTsumo(score.base(), seat, pao, state.riichiMask());
    end.setScore(0, score);
    if (state.riichi(seat) != RiichiState.NONE) {
      revealUra();
    }
    return finish(ws);
  }

  /** 直前の打牌・加槓・抜いた北へのロン。和了者は、放銃した席から見て手番の近い順に並ぶ（他家の応答と局の結果で同じ順）。 */
  private Boundary ron(RuleWorkspace ws) {
    int loser = state.last().seat();
    boolean riichi = false;
    for (int i = 0; i < window.winnerCount(); i++) {
      int winner = window.winner(i);
      WinScore score = decisions.win(decisions.slotOf(winner));
      int pao = WinLegality.pao(full.seat(winner), score, loser);
      if (i == 0) {
        end.outcome().setRon(score.base(), winner, loser, pao, state.riichiMask());
      } else {
        end.outcome().addRonWinner(score.base(), winner, pao);
      }
      end.setScore(i, score);
      riichi |= state.riichi(winner) != RiichiState.NONE;
    }
    if (riichi) {
      revealUra();
    }
    return finish(ws);
  }

  private Boundary exhaustive(RuleWorkspace ws) {
    EventLog events = state.events();
    int tenpai = 0;
    int nagashi = 0;
    for (int seat = 0; seat < seats; seat++) {
      if (WinLegality.tenpai(hands[seat])) {
        tenpai |= 1 << seat;
      }
      if (WinLegality.nagashi(events, seat)) {
        nagashi |= 1 << seat;
      }
    }
    end.outcome().setExhaustive(tenpai, nagashi, state.riichiMask());
    return finish(ws);
  }

  private Boundary abortive(AbortiveDraw draw, RuleWorkspace ws) {
    end.outcome().setAbortive(draw, state.riichiMask());
    return finish(ws);
  }

  /** 局を終える。各席の待ちを局末に書き、判断を空にする。 */
  private Boundary finish(RuleWorkspace ws) {
    for (int seat = 0; seat < seats; seat++) {
      end.setWaits(seat, waitsAtEnd(seat, ws));
    }
    decisions.clear();
    return Boundary.ROUND_END;
  }

  /** 局末の形の上の和了牌。13 枚相当の手はその待ち。ツモの後の 14 枚相当の手は、直前にツモった種類を除いた 13 枚の待ち。 */
  private long waitsAtEnd(int seat, RuleWorkspace ws) {
    Hand hand = hands[seat];
    if (hand.shape().size() == Variant.HAND) {
      return hand.waits();
    }
    HandShape probe = ws.probe();
    probe.copyFrom(hand.shape());
    probe.remove(state.last().tile());
    return ws.analyzer().waits(probe);
  }

  /** 判断が持つ和了の点を、いまの牌山の裏ドラで求め直す。和了できなくなっていれば IllegalStateException を投げる。 */
  private void rescoreWin(int slot, int seat, RuleWorkspace ws) {
    WinScore score = decisions.win(slot);
    TurnEvent last = state.last();
    boolean won =
        decisions.kind(slot) == EngineDecisionKind.TURN
            ? WinLegality.tsumo(state, hands[seat], wall, ws, score)
            : WinLegality.ron(
                state, hands[seat], seat, last.tile(), last.red(), Window.ronSource(state), wall, ws, score);
    if (!won) {
      throw new IllegalStateException("Seat " + seat + " lost its win after the wall was reshuffled");
    }
  }

  private void revealPending() {
    if (state.dora().pending()) {
      revealIndicator();
    }
  }

  /** 牌山の次の表示牌をめくり、偶然のイベントに足す。 */
  private void revealIndicator() {
    int tile = wall.indicator(state.dora().count());
    state.reveal(tile);
    chance.appendIndicator(tile);
  }

  /** 立直の和了で局を終えるときに、めくった表示牌ごとに、対応する裏ドラ表示牌を偶然のイベントに足す。 */
  private void revealUra() {
    for (int index = 0; index < state.dora().count(); index++) {
      chance.appendUra(wall.ura(index));
    }
  }
}
