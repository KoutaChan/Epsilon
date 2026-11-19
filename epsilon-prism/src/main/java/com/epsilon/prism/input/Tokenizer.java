package com.epsilon.prism.input;

import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.Furiten;
import com.epsilon.prism.input.InputSchema.RiichiExact;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.TransitionScalar;
import com.epsilon.prism.input.InputSchema.WaitClass;
import com.epsilon.prism.input.InputSchema.WinExact;
import com.epsilon.prism.input.InputSchema.Wind;
import com.epsilon.prism.input.RowInputWriter.TokenWriter;
import com.epsilon.prism.input.WaitScoring.Stance;
import java.util.Arrays;
import java.util.Objects;

/** 判断の観測から、ルールに応じた局・席・牌・イベント・合法手と遷移の入力を書く。 */
public final class Tokenizer implements InputFormat.RowWriter {
  private final InputFormat schema;
  private final RowInputWriter writer;
  private final RowFacts facts;

  private static final Wind[] WINDS = Wind.values();
  private static final ScoreLadder.Pass[] PASSES = ScoreLadder.Pass.values();
  private static final SeatField[] PASS_FIELDS = passFields();
  private static final int NONE = -1;

  private static final DiscardExact[] DISCARD_EXACTS = DiscardExact.values();
  private final WinExact[] winFields;
  private static final RiichiExact[] RIICHI_EXACTS = RiichiExact.values();

  private final int[] firstTransition = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorShanten = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorUkeire = new int[Limits.MAX_LEGAL_ACTIONS];
  private final float[] riichiExact = new float[Limits.MAX_LEGAL_ACTIONS * RIICHI_EXACTS.length];
  private final float[] declared = new float[DISCARD_EXACTS.length];

  private boolean waitOffered;
  private boolean tsumoLegal;
  private int passAction;

  public Tokenizer(Variant variant) {
    this(InputSchema.of(variant));
  }

  Tokenizer(InputFormat schema) {
    this.schema = schema;
    writer = new RowInputWriter(schema);
    winFields = schema.winFields().constants().toArray(WinExact[]::new);
    facts = new RowFacts(schema);
  }

  @Override
  public void write(Observation o, HostInputs batch, int row) {
    write(o, null, batch, row);
  }

  @Override
  public void write(Observation o, EventHistory history, HostInputs batch, int row) {
    RowInputWriter out = writer.bind(batch.input(), row);
    var prefix = batch.input().publication(row);
    if (prefix != null) {
      if (prefix.records != o.events() || prefix.end != o.publicEvents()) {
        throw new IllegalStateException("event publication does not match the observation");
      }
      var previous = EventHistory.Bytes.find(schema, prefix);
      long extent = previous == null ? 0 : previous.extent();
      facts.load(o, (int) (extent >>> Integer.SIZE));
      facts.fromPosition = (int) extent;
      out.writePublished(facts, previous, prefix);
      writeRest(o, out);
    } else if (history != null && history.acquire(schema)) {
      try {
        facts.load(o, history.match(o));
        history.settle(o, facts.settledRecords);
        out.restoreEvents(history, facts.fromPosition);
        out.writeColumns(facts);
        out.saveEvents(history, facts.settledPositions);
        writeRest(o, out);
      } finally {
        history.release();
      }
    } else {
      write(o, out);
    }
  }

  private void write(Observation o, RowInputWriter out) {
    facts.load(o);
    out.writeColumns(facts);
    writeRest(o, out);
  }

  /** 列の後の、席・遷移・手を書いて行を写す。 */
  private void writeRest(Observation o, RowInputWriter out) {
    writeSeats(o, out);
    out.prepareCodes(facts);
    writeActions(o, out);
    out.commit(o.decisionKind());
  }

  /**
   * 直前に {@link #write} した判断の記録と同じ色の正準化で、物理の牌を正準化した牌のトークンに変える。補助のヘッドの教師の牌の軸に、入力と同じ
   * 正準化をかけるために使う。
   *
   * @param physicalTile 物理の牌の種類
   * @return 正準化した牌のトークンの番号
   * @throws IllegalStateException 三人打ちで使わない牌の種類のとき
   */
  @Override
  public int canonicalTile(int physicalTile) {
    return facts.token(Objects.checkIndex(physicalTile, Tiles.KINDS));
  }

  private void writeSeats(Observation o, RowInputWriter out) {
    for (int seat = 0; seat < facts.seatCount; seat++) {
      TokenWriter<SeatField> token = out.seat(seat);
      token.id(SeatField.RELATIVE_SEAT, seat);
      token.value(SeatField.WIND, WINDS[Math.floorMod(seat - o.dealer(), facts.seatCount)]);
      token.id(SeatField.RANK, facts.seatFacts.rank(seat));
      token.value(SeatField.RIICHI, o.riichi(seat));
      token.flag(SeatField.IPPATSU, o.ippatsu(seat));
      token.flag(SeatField.MENZEN, facts.seatFacts.openMelds(seat) == 0);
      token.flag(SeatField.BEFORE_FIRST_DISCARD, o.beforeFirstDiscard(seat));
      token.count(SeatField.MELDS, facts.seatFacts.meldCount(seat));
      token.count(SeatField.KANS, facts.seatFacts.kans(seat));
      if (facts.nuki) token.count(SeatField.NUKI, o.nuki(seat));
      if (facts.chi) token.flag(SeatField.OPEN_TRIPLETS, facts.seats.triplets(seat));
      token.count(SeatField.DISCARDS, facts.events.discards(seat));
      token.count(SeatField.OPEN_YAKUHAI_HAN, facts.seats.yakuhaiHan(seat));
      token.value(SeatField.OPEN_SUIT, facts.openSuit(facts.seats.openSuit(seat)));
      token.flag(SeatField.OPEN_TANYAO, facts.seats.tanyao(seat));
      token.count(SeatField.OPEN_DORA, facts.seats.openDora(seat));
      token.flag(SeatField.NAGASHI, facts.seats.nagashi(seat));
      if (seat != 0) {
        for (ScoreLadder.Pass pass : PASSES) {
          token.value(PASS_FIELDS[pass.ordinal()], facts.ladder.grade(seat, pass));
        }
      }
    }
  }

  private static SeatField[] passFields() {
    SeatField[] fields = new SeatField[PASSES.length];
    for (ScoreLadder.Pass pass : PASSES) {
      fields[pass.ordinal()] =
          switch (pass) {
            case BY_DIRECT_RON -> SeatField.PASS_BY_DIRECT_RON;
            case BY_OTHER_RON -> SeatField.PASS_BY_OTHER_RON;
            case BY_TSUMO -> SeatField.PASS_BY_TSUMO;
            case PASSED_BY_RON -> SeatField.PASSED_BY_RON;
            case PASSED_BY_TSUMO -> SeatField.PASSED_BY_TSUMO;
          };
    }
    return fields;
  }

  private void writeActions(Observation o, RowInputWriter out) {
    int actions = o.legalActions();
    tsumoLegal = false;
    passAction = NONE;
    for (int action = 0; action < actions; action++) {
      Action.Type type = o.action(action).type();
      tsumoLegal |= type == Action.Type.TSUMO_AGARI;
      if (type == Action.Type.PASS) {
        passAction = action;
      }
    }
    EngineDecisionKind kind = o.decisionKind();
    boolean response =
        kind == EngineDecisionKind.RESPONSE
            || kind == EngineDecisionKind.CHANKAN
            || kind == EngineDecisionKind.NUKI_RESPONSE;
    waitOffered = response && (facts.currentWaits & Tiles.bit(o.lastEventTile())) != 0L;
    int transition = 0;
    for (int action = 0; action < actions; action++) {
      Action chosen = o.action(action);
      int representative = o.representative(action);
      int count = representative == action ? o.transitionsOf(action) : 0;
      if (representative == action) {
        firstTransition[action] = transition;
        for (int index = 0; index < count; index++) {
          writeTransition(o, out, action, chosen, index, transition++);
        }
      }
      ActionRole role = ActionRole.of(chosen.type());
      int base = response && action != passAction ? passAction : NONE;
      if (chosen.type() == Action.Type.RIICHI_DAHAI) {
        base = damaOf(o, chosen);
      }
      out.action(action, role, firstTransition[representative], count, base, representative);
      writeActionExact(o, out, action, chosen, representative);
      if (role == ActionRole.DISCARD || role == ActionRole.RIICHI) {
        out.prior(
            action,
            PriorFeatures.shanten(priorShanten[representative]),
            PriorFeatures.ukeire(priorUkeire[representative]));
      }
    }
  }

  private void writeActionExact(
      Observation o, RowInputWriter out, int action, Action chosen, int representative) {
    switch (chosen.type()) {
      case RON_AGARI, TSUMO_AGARI -> {
        facts.wins.load(o, o.winOf(action));
        for (WinExact field : winFields) {
          out.exact(action, field, field.value(facts.wins));
        }
      }
      case RIICHI_DAHAI -> {
        int at = representative * RIICHI_EXACTS.length;
        for (RiichiExact field : RIICHI_EXACTS) {
          out.exact(action, field, riichiExact[at + field.ordinal()]);
        }
      }
      default -> {}
    }
  }

  /** 立直の打牌の基準になるダマの打牌を返す（同じ牌を、同じ赤・ツモ切りの選び方で切る DAHAI）。 */
  private static int damaOf(Observation o, Action riichi) {
    int dama = Action.code(Action.Type.DAHAI, riichi.tile(), riichi.selection());
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (o.action(action).code() == dama) {
        return action;
      }
    }
    throw new IllegalStateException("riichi discard has no dama discard: " + riichi);
  }

  private void writeTransition(
      Observation o, RowInputWriter out, int action, Action chosen, int index, int transition) {
    Action.Type type = chosen.type();
    AfterHand after = facts.hands.move(action, index);
    Action discard = facts.hands.discard();
    int discarded = discard == null ? NONE : discard.tile();
    TokenWriter<TransitionField> field = out.transition(transition);
    field.value(TransitionField.ACTION, type);
    field.id(
        TransitionField.MAIN_TILE,
        facts.token(type == Action.Type.NUKI ? Tiles.PEI : chosen.tile()));
    field.value(TransitionField.TILE_SELECTION, chosen.selection());
    if (discard != chosen && discard != null) {
      field.id(TransitionField.CALL_DISCARD, facts.token(discarded));
      field.flag(TransitionField.CALL_DISCARD_AKA, discard.selection().red());
    }
    field.flag(TransitionField.MENZEN, after.isMenzen());
    field.count(TransitionField.DORA, after.dora());
    long ukeire = 0L;
    long waitMask = 0L;
    Stance stance = Stance.DAMA;
    if (TransitionHands.terminal(type)) {
      writeWin(o, field, action, type);
    } else {
      facts.shapes.analyze(after, facts.tiles, facts.hands.slot());
      ukeire = facts.shapes.ukeire();
      waitMask = facts.shapes.waits();
      field.id(TransitionField.SHANTEN, facts.shapes.minimum());
      field.id(TransitionField.STANDARD_SHANTEN, facts.shapes.standard());
      if (facts.shapes.special()) {
        field.id(TransitionField.CHIITOI_SHANTEN, facts.shapes.chiitoi());
        field.id(TransitionField.KOKUSHI_SHANTEN, facts.shapes.kokushi());
      }
      field.count(TransitionField.UKEIRE_KINDS, Long.bitCount(ukeire));
      field.count(TransitionField.WAIT_KINDS, Long.bitCount(waitMask));
      Furiten furiten = furiten(type, discard, waitMask);
      field.value(TransitionField.FURITEN, furiten);
      int live = facts.shapes.liveUkeire();
      out.scalar(transition, TransitionScalar.LIVE_UKEIRE, live);
      out.scalar(transition, TransitionScalar.HIT_WITHIN_1, facts.shapes.hitWithin(1));
      out.scalar(transition, TransitionScalar.HIT_WITHIN_2, facts.shapes.hitWithin(2));
      out.scalar(transition, TransitionScalar.HIT_WITHIN_3, facts.shapes.hitWithin(3));
      if (index == 0) {
        priorShanten[action] = facts.shapes.minimum();
        priorUkeire[action] = live;
      }
      if (waitMask != 0L) {
        RiichiState riichi = type == Action.Type.RIICHI_DAHAI ? o.declaration() : facts.selfRiichi;
        stance = riichi == RiichiState.NONE ? Stance.DAMA : Stance.RIICHI;
        facts.waits.score(
            after, waitMask, facts.tiles, o, PublicSeatFacts.seatWindTile(o, 0), riichi);
        out.scalar(transition, TransitionScalar.LIVE_RON_YAKU_WAITS, liveYakuWaits(Method.RON));
        out.scalar(transition, TransitionScalar.LIVE_TSUMO_YAKU_WAITS, liveYakuWaits(Method.TSUMO));
        boolean ronBlocked = furiten == Furiten.DISCARD || furiten == Furiten.RIICHI_PERSISTENT;
        boolean declares = type == Action.Type.RIICHI_DAHAI;
        var projection = facts.projection();
        projection.project(facts.waits, stance, declares, ronBlocked, facts.tiles, waitMask);
        for (DiscardExact exact : DISCARD_EXACTS) {
          float value = projection.value(exact.projected());
          out.exact(transition, exact, value);
          declared[exact.ordinal()] = value;
        }
        if (declares) {
          projection.project(facts.waits, Stance.DAMA, false, ronBlocked, facts.tiles, waitMask);
          int at = action * RIICHI_EXACTS.length;
          for (RiichiExact exact : RIICHI_EXACTS) {
            riichiExact[at + exact.ordinal()] =
                declared[exact.source().ordinal()] - projection.value(exact.source().projected());
          }
        }
      } else if (type == Action.Type.RIICHI_DAHAI) {
        // エンジンは形の上で聴牌なら立直を許す。待ちが全て 5 枚目の牌（4 枚使いの単騎など）なら和了できないので、射影も差も 0
        int at = action * RIICHI_EXACTS.length;
        Arrays.fill(riichiExact, at, at + RIICHI_EXACTS.length, 0f);
      }
    }
    writeCodes(o, out, after, transition, type, discarded, ukeire, waitMask, stance);
  }

  /** 和了の遷移のフィールドを、記録にあるこの手で和了したときの値から書く。九種九牌では何も書かない。 */
  private static void writeWin(
      Observation o, TokenWriter<TransitionField> field, int action, Action.Type type) {
    if (type == Action.Type.KYUSHU_KYUHAI) {
      return;
    }
    int win = o.winOf(action);
    field.count(TransitionField.WIN_HAN, Ids.saturated(o.winHan(win), InputSchema.HAN_SATURATION));
    field.count(TransitionField.WIN_FU, o.winFuCode(win));
    field.count(TransitionField.WIN_YAKUMAN, o.winYakuman(win));
    field.count(
        TransitionField.URA_INDICATORS, o.winUraEligible(win) ? o.winUraIndicators(win) : 0);
  }

  /**
   * 遷移の後の振聴を返す（振聴でなければ null）。待ちが自分の河（この手で捨てる牌を含む）にあれば捨て牌の振聴。自分が打牌する遷移では同巡の
   * 見逃しが消え、立直後の見逃しだけが残る。打牌しない遷移（見送り・槓・北抜き）では、今の見逃しが続く。応答する他家の牌が今の手の形の上の
   * 待ちに入るときも、見逃しの振聴になる（役が無くてロンできない牌も含む）。立直中にツモ和了できる北を抜いても、立直後の見逃しになる。
   */
  private Furiten furiten(Action.Type type, Action discard, long waitMask) {
    long river = facts.safety.river(0);
    if (discard != null) {
      river |= Tiles.bit(discard.tile());
    }
    if ((waitMask & river) != 0L) {
      return Furiten.DISCARD;
    }
    boolean riichi = facts.selfRiichi != RiichiState.NONE;
    if (discard != null) {
      return riichi && facts.furitenTemporary ? Furiten.RIICHI_PERSISTENT : null;
    }
    if (facts.furitenTemporary || waitOffered || riichi && tsumoLegal && type == Action.Type.NUKI) {
      return riichi ? Furiten.RIICHI_PERSISTENT : Furiten.TEMPORARY;
    }
    return null;
  }

  /** 立直しなくても役がつく待ち牌の、生きている枚数。 */
  private int liveYakuWaits(Method method) {
    long yaku = 0L;
    for (int wait = 0, count = facts.waits.waits(); wait < count; wait++) {
      if (facts.waits.available(wait, Stance.DAMA, method, WinningTile.PLAIN)) {
        yaku |= Tiles.bit(facts.waits.waitTile(wait));
      }
    }
    return facts.tiles.live(yaku);
  }

  /** 遷移のキーのコードを、局 1・席 3・牌 27 の領域ごとにまとめて書く。 */
  private void writeCodes(
      Observation o,
      RowInputWriter out,
      AfterHand after,
      int transition,
      Action.Type type,
      int discarded,
      long ukeire,
      long waitMask,
      Stance stance) {
    int source =
        switch (type) {
          case CHI, PON, DAIMINKAN, RON_AGARI -> o.lastEventSeat();
          default -> NONE;
        };
    out.startCodes(transition, source);
    // 判断時の手牌を下書きに使い、遷移で変わる牌の桁を書く。
    long tilesWithCode = after.removedKinds() | ukeire | waitMask;
    for (long rest = tilesWithCode; rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      long bit = 1L << tile;
      WaitClass wait = (waitMask & bit) != 0L ? waitClass(tile, stance) : null;
      out.tileCode(
          facts.token(tile),
          schema.tileCode(after.count(tile), (ukeire & bit) != 0L, wait, tile == discarded));
    }
  }

  /**
   * 待ちの分類を返す。その遷移の立直の状態で、赤でない和了牌でロンしたときの翻で分ける。ロンに役がなくツモにあればツモだけ、どちらにも
   * なければ待ちでない（null）。
   */
  private WaitClass waitClass(int tile, Stance stance) {
    for (int wait = 0, count = facts.waits.waits(); wait < count; wait++) {
      if (facts.waits.waitTile(wait) != tile) {
        continue;
      }
      if (facts.waits.available(wait, stance, Method.RON, WinningTile.PLAIN)) {
        int han = facts.waits.han(wait, stance, Method.RON, WinningTile.PLAIN);
        if (han == 0 || han >= 6) {
          return WaitClass.HAN_6_PLUS;
        }
        return han >= 3 ? WaitClass.HAN_3_5 : WaitClass.HAN_1_2;
      }
      return facts.waits.available(wait, stance, Method.TSUMO, WinningTile.PLAIN)
          ? WaitClass.TSUMO_ONLY
          : null;
    }
    throw new IllegalStateException("tile " + tile + " is not a scored wait");
  }
}
