package com.epsilon.prism3p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.PublicSeatFacts;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.input.TransitionHands;
import com.epsilon.prism.input.WaitProjection;
import com.epsilon.prism.input.WaitScoring;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism3p.input.InputSchema.ActionRole;
import com.epsilon.prism3p.input.InputSchema.DiscardExact;
import com.epsilon.prism3p.input.InputSchema.EventField;
import com.epsilon.prism3p.input.InputSchema.OpenSuit;
import com.epsilon.prism3p.input.InputSchema.RoundField;
import com.epsilon.prism3p.input.InputSchema.SeatField;
import com.epsilon.prism3p.input.InputSchema.TileField;
import com.epsilon.prism3p.input.InputSchema.TransitionField;
import com.epsilon.prism3p.input.InputSchema.WinExact;
import com.epsilon.prism3p.input.InputSchema.Wind;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 手で作った局面で、三人打ちの入力を手で計算した値と照合する。見るのは、トークンの形、抜いた北の数とドラの数、北を抜く手と抜いた後の遷移、
 * 抜いた北への応答、西入の見込み、筒子と索子だけの色の入れ替え、副露の色、鳴きはポンだけで遷移が鳴いた後の打牌ごとになること（チーは無い）、残りのツモの数、
 * ツモ損の点の見積もり（抜きドラの翻を含む）、オーラスの効用 [2, 0, −2]、北を抜いた後の立直がダブル立直にならないこと。
 */
public class HandBuiltPositionTest {

  private static final TileTokens TOKENS = TileTokens.of(TENHOU_3P);
  private static final int NORTH_TOKEN = TOKENS.token(Tiles.PEI);
  private static final int WEST_TOKEN = TOKENS.token(Tiles.SHA);

  /** 北を持たない不聴の手。 */
  private static final String NOTEN_A = "19m258p147s369s67z";

  private static final String NOTEN_B = "19m369p258s147p57z";
  private static final String NOTEN_C = "19m369p258s369s57z";

  private final RowEncoder r = new RowEncoder();

  private Observation at(PlayedScene scene, int seat) {
    return r.encode(scene.engine, scene.engine.decisions().slotOf(seat));
  }

  private static int action(Observation o, Action.Type type, int tile) {
    for (int index = 0; index < o.legalActions(); index++) {
      Action action = o.action(index);
      if (action.type() == type && (tile < 0 || action.tile() == tile)) {
        return index;
      }
    }
    throw new AssertionError("no " + type + " of tile " + tile);
  }

  private static int action(Observation o, Action.Type type) {
    return action(o, type, -1);
  }

  @Test
  public void theTokensHaveTheThreePlayerShape() {
    PlayedScene scene = PlayedScene.first().hand(0, NOTEN_A).hand(1, NOTEN_B).hand(2, NOTEN_C).draws("1z").indicator("1z").start();
    Observation o = at(scene, 0);
    int hand = 0;
    for (int token = 0; token < TOKENS.count(); token++) {
      assertEquals(r.tile(token, TileField.TILE), token + 1, "token " + token);
      int held = r.tile(token, TileField.HAND);
      hand += held;
      assertEquals(held + r.tile(token, TileField.VISIBLE) + r.tile(token, TileField.UNSEEN), Tiles.COPIES, "token " + token);
    }
    assertEquals(hand, 14);
    assertEquals(r.tile(TOKENS.token(Tiles.TON), TileField.VISIBLE), 1, "the indicator is visible");
    for (int seat = 0; seat < 3; seat++) {
      assertEquals(r.seat(seat, SeatField.RELATIVE_SEAT), seat + 1);
      assertEquals(r.seat(seat, SeatField.WIND), RowEncoder.id(SeatField.WIND, Wind.values()[seat]));
      assertEquals(r.seat(seat, SeatField.NUKI), 0);
    }
    assertEquals(r.round(RoundField.DECISION_KIND), RowEncoder.id(RoundField.DECISION_KIND, EngineDecisionKind.TURN));
    assertEquals(r.round(RoundField.ROUND_WIND), RowEncoder.id(RoundField.ROUND_WIND, Wind.EAST));
    assertEquals(r.round(RoundField.KYOKU), 1);
    assertEquals(o.remainingWall(), 54);
    assertEquals(r.round(RoundField.DRAWS_LEFT), 18, "the dealer draws 18 more times from a wall of 54");
  }

  /**
   * 表示牌が西なら北が表ドラになる。北のトークンのドラの数は、表示牌 1 枚分の 1 と抜きドラの 1 で 2。北を 2 枚持つ手の遷移のドラは、
   * ほかの牌を打つと 2（手の北 2 枚の表ドラ）、北を抜くと 3（抜いた北 1 枚の 2 と、手に残る北の 1）。北を抜くと、席の抜きの数が増え、
   * 公開で見える枚数に入り、イベントに NUKI が足され、補充は嶺上の扱いになる。
   */
  @Test
  public void aNorthIsOneDoraAndIsTakenOutAsAnEvent() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "123p456p789s11s4z4z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("5z")
            .rinshan("2p")
            .indicator("3z")
            .start();
    Observation before = at(scene, 0);
    assertEquals(r.tile(NORTH_TOKEN, TileField.DORA), 2, "one indicator and the nuki dora");
    assertEquals(r.tile(WEST_TOKEN, TileField.DORA), 0);
    assertEquals(r.tile(NORTH_TOKEN, TileField.HAND), 2);
    assertEquals(r.tile(NORTH_TOKEN, TileField.VISIBLE), 0);
    int nuki = action(before, Action.Type.NUKI);
    assertEquals(r.column(nuki, ActionColumn.ROLE), ActionRole.NUKI.ordinal());
    assertEquals(r.column(nuki, ActionColumn.BASE), -1);
    assertEquals(r.column(nuki, ActionColumn.TRANSITIONS), 1);
    int transition = r.column(nuki, ActionColumn.FIRST_TRANSITION);
    assertEquals(r.transition(transition, TransitionField.ACTION), RowEncoder.id(TransitionField.ACTION, Action.Type.NUKI));
    assertEquals(r.transition(transition, TransitionField.MAIN_TILE), NORTH_TOKEN + 1);
    assertEquals(r.transition(transition, TransitionField.DORA), 3);
    int discard = action(before, Action.Type.DAHAI, Tiles.HAKU);
    assertEquals(r.transition(r.column(discard, ActionColumn.FIRST_TRANSITION), TransitionField.DORA), 2);

    scene.play("0:NUKI");
    assertTrue(scene.turnOf(0));
    Observation after = at(scene, 0);
    assertEquals(after.remainingWall(), 53, "the replacement shrinks the live wall");
    assertEquals(r.seat(0, SeatField.NUKI), 1);
    assertEquals(r.tile(NORTH_TOKEN, TileField.HAND), 1);
    assertEquals(r.tile(NORTH_TOKEN, TileField.VISIBLE), 1);
    assertEquals(r.tile(NORTH_TOKEN, TileField.UNSEEN), 2);
    assertEquals(r.event(0, EventField.KIND), RowEncoder.id(EventField.KIND, PublicEvent.NUKI));
    assertEquals(r.event(0, EventField.AUTHOR), 1);
    assertEquals(r.event(0, EventField.TILE), NORTH_TOKEN + 1);
    assertEquals(r.event(0, EventField.DISCARD_INDEX), 0, "a nuki is not a discard");
    assertEquals(r.round(RoundField.LAST_EVENT_RINSHAN), 1);
    assertEquals(r.round(RoundField.CALL_MADE), 0, "a nuki is not a call");

    // 次の席から見ると、抜いた席は上家（相対席 2）。
    scene.passUntil(s -> s.turnOf(1));
    at(scene, 1);
    assertEquals(r.seat(2, SeatField.NUKI), 1);
    assertEquals(r.event(0, EventField.KIND), RowEncoder.id(EventField.KIND, PublicEvent.NUKI));
    assertEquals(r.event(0, EventField.AUTHOR), 3);
    assertEquals(r.event(1, EventField.KIND), RowEncoder.id(EventField.KIND, PublicEvent.DISCARD));
  }

  /** 西入の見込みが立つのは、南 3 局から西 2 局までで、トップが 40,000 点未満のとき。 */
  @Test
  public void westPossibleFollowsTheFortyThousandLine() {
    assertWestPossible(5, true, 39_000, 33_000, 33_000);
    assertWestPossible(5, false, 40_000, 35_000, 30_000);
    assertWestPossible(7, true, 39_000, 33_000, 33_000);
    assertWestPossible(4, false, 39_000, 33_000, 33_000);
    assertWestPossible(8, false, 39_000, 33_000, 33_000);
  }

  private void assertWestPossible(int round, boolean expected, int... scores) {
    PlayedScene scene =
        PlayedScene.at(round, 0, 0, scores).hand(0, NOTEN_A).hand(1, NOTEN_B).hand(2, NOTEN_C).draws("1z").start();
    at(scene, round % 3);
    assertEquals(r.round(RoundField.KYOKU), round + 1);
    assertEquals(r.round(RoundField.WEST_POSSIBLE), expected ? 1 : 0, "round " + round);
  }

  /** 筒子と索子は、手に多いほうの色が正準の第 1 の色になり、萬子と字牌は動かない。入れ替えた対の手で、牌のトークンはビットで同じになる。 */
  @Test
  public void onlyPinAndSouAreSwapped() {
    PlayedScene sou = PlayedScene.first().hand(0, "123456789s11p4z5z").hand(1, NOTEN_A).hand(2, NOTEN_C).draws("6z").indicator("1z").start();
    PlayedScene pin = PlayedScene.first().hand(0, "123456789p11s4z5z").hand(1, NOTEN_A).hand(2, NOTEN_C).draws("6z").indicator("1z").start();
    RowEncoder other = new RowEncoder();
    r.encode(sou.engine, 0);
    other.encode(pin.engine, 0);
    assertEquals(r.tokenizer.canonicalTile(Tiles.S1), TOKENS.token(Tiles.P1), "the longer suit comes first");
    assertEquals(r.tokenizer.canonicalTile(Tiles.P1), TOKENS.token(Tiles.S1));
    assertEquals(other.tokenizer.canonicalTile(Tiles.P1), TOKENS.token(Tiles.P1));
    assertEquals(r.tokenizer.canonicalTile(Tiles.M1), 0, "manzu does not move");
    assertEquals(r.tokenizer.canonicalTile(Tiles.M9), 1);
    assertEquals(r.tokenizer.canonicalTile(Tiles.TON), TOKENS.token(Tiles.TON));
    for (Tensor tensor : List.of(Tensor.ROUND_CATEGORY, Tensor.SEAT_CATEGORY, Tensor.TILE_CATEGORY)) {
      int count = r.batch.plan().count(tensor.axis());
      assertTrue(Arrays.equals(r.bytes(tensor, 0, count), other.bytes(tensor, 0, count)), tensor.name());
    }
  }

  /** ポンは RESPONSE の判断で、手の扱いは CALL、基準の手は見送り、遷移は鳴いた後の打牌ごとに作る。三人打ちにチーは無い。副露の色は、鳴いた席のトークンに出る。 */
  @Test
  public void ponIsTheOnlyCallAndShowsTheOpenSuit() {
    PlayedScene manzu = PlayedScene.first().hand(0, NOTEN_A).hand(1, "11m369p258s147p5z6z").hand(2, NOTEN_C).draws("4z").start();
    manzu.play("0:DAHAI(1m)");
    assertEquals(manzu.legal(1), List.of("PON(1m)", "PASS"));
    Observation o = at(manzu, 1);
    assertEquals(o.decisionKind(), EngineDecisionKind.RESPONSE);
    int pon = action(o, Action.Type.PON);
    int pass = action(o, Action.Type.PASS);
    assertEquals(r.column(pon, ActionColumn.ROLE), ActionRole.CALL.ordinal());
    assertEquals(r.column(pon, ActionColumn.BASE), pass);
    assertEquals(r.column(pon, ActionColumn.TRANSITIONS), o.callDiscardsOf(pon));
    assertEquals(r.column(pon, ActionColumn.TRANSITIONS), 11);
    int first = r.column(pon, ActionColumn.FIRST_TRANSITION);
    for (int index = 0; index < o.callDiscardsOf(pon); index++) {
      assertEquals(r.transition(first + index, TransitionField.MAIN_TILE), TOKENS.token(Tiles.M1) + 1);
      assertTrue(r.transition(first + index, TransitionField.CALL_DISCARD) > 0);
    }
    manzu.play("1:PON(1m)");
    Observation posted = at(manzu, 1);
    assertEquals(posted.decisionKind(), EngineDecisionKind.POST_CALL_DAHAI);
    assertEquals(r.seat(0, SeatField.OPEN_SUIT), RowEncoder.id(SeatField.OPEN_SUIT, OpenSuit.MANZU));
    assertEquals(r.seat(0, SeatField.MELDS), 1);
    assertEquals(r.seat(0, SeatField.MENZEN), 0);

    PlayedScene honors = PlayedScene.first().hand(0, NOTEN_A).hand(1, "55z19m369p258s147p").hand(2, NOTEN_C).draws("5z").start();
    honors.play("0:DAHAI_TSUMOGIRI(P)");
    honors.play("1:PON(P)");
    at(honors, 1);
    assertEquals(r.seat(0, SeatField.OPEN_SUIT), RowEncoder.id(SeatField.OPEN_SUIT, OpenSuit.HONORS));

    // 手に多いほうの色が正準の第 1 の色。2p のポンは、筒子の多い手では第 1 の色、索子の多い手では第 2 の色になる。
    PlayedScene pinHeavy = PlayedScene.first().hand(0, NOTEN_A).hand(1, "22p123p456p789p1z5z").hand(2, NOTEN_C).draws("4z").start();
    pinHeavy.play("0:DAHAI(2p)");
    pinHeavy.play("1:PON(2p)");
    at(pinHeavy, 1);
    assertEquals(r.seat(0, SeatField.OPEN_SUIT), RowEncoder.id(SeatField.OPEN_SUIT, OpenSuit.FIRST));
    PlayedScene souHeavy = PlayedScene.first().hand(0, NOTEN_A).hand(1, "22p123s456s789s1z5z").hand(2, NOTEN_C).draws("4z").start();
    souHeavy.play("0:DAHAI(2p)");
    souHeavy.play("1:PON(2p)");
    at(souHeavy, 1);
    assertEquals(r.seat(0, SeatField.OPEN_SUIT), RowEncoder.id(SeatField.OPEN_SUIT, OpenSuit.SECOND));
  }

  /** 全員の手番をツモ切りで最後まで打つ。どの判断でも、残りのツモのフィールドがその後の自分の手番の数と一致する。 */
  @Test
  public void drawsLeftCountsTheRemainingTurns() {
    PlayedScene scene = PlayedScene.first().hand(0, NOTEN_A).hand(1, NOTEN_B).hand(2, NOTEN_C).start();
    Engine engine = scene.engine;
    RuleWorkspace ws = new RuleWorkspace(TENHOU_3P);
    int checked = 0;
    while (!scene.ended()) {
      Decisions decisions = engine.decisions();
      for (int slot = 0; slot < decisions.size(); slot++) {
        int seat = decisions.seat(slot);
        r.encode(engine, slot);
        assertEquals(r.round(RoundField.DRAWS_LEFT), turnsAfterThis(engine, ws, seat), "seat " + seat + " kind " + decisions.kind(slot));
        checked++;
      }
      scene.step();
    }
    assertTrue(checked > 50, "checked " + checked);
  }

  private static int turnsAfterThis(Engine engine, RuleWorkspace ws, int seat) {
    Engine copy = engine.copy();
    Boundary boundary = PlayedScene.passOrTsumogiri(copy, ws);
    int turns = 0;
    while (boundary == Boundary.DECISIONS) {
      Decisions decisions = copy.decisions();
      if (decisions.kind(0) == EngineDecisionKind.TURN && decisions.seat(0) == seat) {
        turns++;
      }
      boundary = PlayedScene.passOrTsumogiri(copy, ws);
    }
    return turns;
  }

  /**
   * ツモ損の確かめ。親のメンゼンツモ＋一通の 3 翻 30 符（基本点 960）は 2,000 点オールで計 4,000 点（四人打ちなら 6,000 点）、子は
   * 2,000＋1,000 で 3,000 点。抜いた北 1 枚で 4 翻になると、親は 3,900 点オールで計 7,800 点。
   */
  @Test
  public void tsumoPointsAreTheThreePlayerSettlement() {
    PlayedScene dealer =
        PlayedScene.first()
            .hand(0, "123p456p789p11s3z3z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("7z7s9s1s")
            .indicator("1m")
            .start()
            .passUntil(s -> s.turnOf(0) && s.legal(0).contains("TSUMO_AGARI"));
    assertTsumoPoints(dealer, 0, 4_000);

    PlayedScene child =
        PlayedScene.first()
            .hand(0, NOTEN_A)
            .hand(1, "123p456p789p11s3z3z")
            .hand(2, NOTEN_C)
            .draws("7z7s9s8p1s")
            .indicator("1m")
            .start()
            .passUntil(s -> s.turnOf(1) && s.legal(1).contains("TSUMO_AGARI"));
    assertTsumoPoints(child, 1, 3_000);

    PlayedScene nuki =
        PlayedScene.first()
            .hand(0, "123p456p789p11s3z4z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("3z7s9s1s")
            .rinshan("7z")
            .indicator("1m")
            .start();
    nuki.play("0:NUKI");
    nuki.passUntil(s -> s.turnOf(0) && s.legal(0).contains("TSUMO_AGARI"));
    assertTsumoPoints(nuki, 0, 7_800);
  }

  private void assertTsumoPoints(PlayedScene scene, int seat, int points) {
    Observation o = at(scene, seat);
    int tsumo = action(o, Action.Type.TSUMO_AGARI);
    assertEquals(r.column(tsumo, ActionColumn.ROLE), ActionRole.TSUMO.ordinal());
    assertEquals(r.winExact(tsumo, WinExact.POINTS), points / 10_000f);
    assertEquals(r.winExact(tsumo, WinExact.ENDS), 0f);
    assertEquals(r.winExact(tsumo, WinExact.RANK_1), 1f, "the winner leads");
  }

  /** 西 3 局は必ず半荘が終わるので、トップがロン和了したときの効用は三人打ちの 1 位の +2。点は 2 翻 40 符の子のロンで 2,600 点。 */
  @Test
  public void theFinalRoundUsesTheThreePlayerUtility() {
    PlayedScene scene =
        PlayedScene.at(8, 0, 0, 40_000, 35_000, 30_000)
            .hand(0, "123p456p789p7s9s3z3z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("8s")
            .indicator("1z")
            .start();
    scene.play("2:DAHAI_TSUMOGIRI(8s)");
    Observation o = at(scene, 0);
    int ron = action(o, Action.Type.RON_AGARI);
    assertEquals(r.column(ron, ActionColumn.ROLE), ActionRole.RON.ordinal());
    assertEquals(r.column(ron, ActionColumn.BASE), action(o, Action.Type.PASS));
    assertEquals(r.winExact(ron, WinExact.ENDS), 1f);
    assertEquals(r.winExact(ron, WinExact.UTILITY), 2f);
    assertEquals(r.winExact(ron, WinExact.RANK_1), 1f);
    assertEquals(r.winExact(ron, WinExact.RANK_2), 0f);
    assertEquals(r.winExact(ron, WinExact.RANK_3), 0f);
    assertEquals(r.winExact(ron, WinExact.POINTS), 2_600 / 10_000f);
  }

  /**
   * 抜いた北への応答は NUKI_RESPONSE の判断になる。ロンの手の扱いは RON、基準の手は見送りで、判断のクラスは NUKI_RESPONSE（南 3 局より前）。
   * 残りのツモは、抜いた席が補充を引いて山が 1 枚減った後で数える（山 51 → 50。自分は抜いた席の下家なので、50 − 1 を 3 で割って 1 を足す）。
   */
  @Test
  public void aNukiCanBeRobbedByTheNukiResponse() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "19m258p369s147s5z6z")
            .hand(1, "123p456p789s11z44z")
            .hand(2, "19m369p258s147s3z7z")
            .draws("9p2z8p4z")
            .rinshan("5s")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9p)");
    scene.play("1:RIICHI_TSUMOGIRI(S)");
    scene.play("2:DAHAI_TSUMOGIRI(8p)");
    scene.play("0:NUKI");
    Observation o = at(scene, 1);
    assertEquals(o.decisionKind(), EngineDecisionKind.NUKI_RESPONSE);
    assertEquals(r.round(RoundField.DECISION_KIND), RowEncoder.id(RoundField.DECISION_KIND, EngineDecisionKind.NUKI_RESPONSE));
    assertEquals(o.remainingWall(), 51);
    assertEquals(r.round(RoundField.DRAWS_LEFT), 17);
    int ron = action(o, Action.Type.RON_AGARI);
    int pass = action(o, Action.Type.PASS);
    assertEquals(r.column(ron, ActionColumn.ROLE), ActionRole.RON.ordinal());
    assertEquals(r.column(ron, ActionColumn.BASE), pass);
    assertEquals(r.column(pass, ActionColumn.ROLE), ActionRole.PASS.ordinal());
    int[] legal = new int[o.legalActions()];
    for (int index = 0; index < legal.length; index++) {
      legal[index] = o.action(index).code();
    }
    SeatFacts facts = new SeatFacts(TENHOU_3P);
    facts.load(o);
    Kind kind = Kind.classify(o.decisionKind(), legal);
    assertEquals(kind, Kind.NUKI_RESPONSE);
    assertEquals(
        DecisionClass.of(TENHOU_3P, kind, Stage.of(TENHOU_3P, o.remainingWall()), Threats.count(o, facts), o.kyoku()),
        DecisionClass.NUKI_RESPONSE);
  }

  /** 北を抜いた後の第一打の立直はダブル立直にならない（エンジンと同じ）。B のヘッドに直接渡す点は、立直 1 翻で見積もる。 */
  @Test
  public void aRiichiAfterANukiIsNotADoubleRiichi() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "123p456p789s11s4z5z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("5z")
            .rinshan("6z")
            .indicator("1m")
            .start();
    scene.play("0:NUKI");
    Observation o = at(scene, 0);
    int riichi = action(o, Action.Type.RIICHI_DAHAI, Tiles.HATSU);
    int transition = r.column(riichi, ActionColumn.FIRST_TRANSITION);

    TileCounts tiles = new TileCounts(TENHOU_3P);
    SeatFacts facts = new SeatFacts(TENHOU_3P);
    tiles.load(o);
    facts.load(o);
    TransitionHands hands = new TransitionHands();
    hands.load(o, tiles, facts);
    ShapeAnalysis shapes = new ShapeAnalysis(TENHOU_3P);
    shapes.reset();
    AfterHand after = hands.move(riichi, 0);
    shapes.analyze(after, tiles, ShapeAnalysis.UNSHARED);
    long waitMask = shapes.waits();
    assertTrue(waitMask != 0L, "the riichi discard leaves a tenpai hand");
    WaitScoring waits = new WaitScoring(TENHOU_3P);
    waits.score(after, waitMask, tiles, o, PublicSeatFacts.seatWindTile(o, 0), RiichiState.RIICHI);
    WaitProjection projection = new WaitProjection(InputSchema.UTILITY);
    projection.load(o);
    projection.project(waits, Stance.RIICHI, true, false, tiles, waitMask);
    assertEquals(
        r.discardExact(transition, DiscardExact.RON_POINTS), projection.value(WaitProjection.Field.RON_POINTS) / 10_000f);
    assertEquals(
        r.discardExact(transition, DiscardExact.TSUMO_POINTS), projection.value(WaitProjection.Field.TSUMO_POINTS) / 10_000f);

    scene.play("0:RIICHI_TSUMOGIRI(F)");
    assertEquals(scene.engine.view(1).riichi(0), RiichiState.RIICHI, "the engine agrees");
  }

  /**
   * 抜いた北 2 枚と手の北の対子で北を 4 枚とも持つと、5 枚目の北は無い。1 索と北のシャンポンの形でも、受け入れと待ちは 1 索だけになる。打牌の後の遷移の
   * フィールドと、その後の 13 枚の手の牌のトークンのフラグの両方で確かめる。
   */
  @Test
  public void aNorthHeldFourTimesWithTheNukiIsNotAWait() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "123p456p789s11s4z4z")
            .hand(1, NOTEN_B)
            .hand(2, NOTEN_C)
            .draws("4z1s")
            .rinshan("4z7z")
            .indicator("1m")
            .start();
    scene.play("0:NUKI");
    scene.play("0:NUKI");
    assertTrue(scene.turnOf(0));
    Observation o = at(scene, 0);
    assertEquals(r.seat(0, SeatField.NUKI), 2);
    assertEquals(r.tile(NORTH_TOKEN, TileField.HAND), 2);
    assertEquals(r.tile(NORTH_TOKEN, TileField.UNSEEN), 0);
    int discard = action(o, Action.Type.DAHAI, Tiles.CHUN);
    int transition = r.column(discard, ActionColumn.FIRST_TRANSITION);
    assertEquals(r.transition(transition, TransitionField.UKEIRE_KINDS), 1, "the fifth north does not exist");
    assertEquals(r.transition(transition, TransitionField.WAIT_KINDS), 1, "the fifth north does not exist");

    TileCounts tiles = new TileCounts(TENHOU_3P);
    SeatFacts facts = new SeatFacts(TENHOU_3P);
    tiles.load(o);
    facts.load(o);
    TransitionHands hands = new TransitionHands();
    hands.load(o, tiles, facts);
    ShapeAnalysis shapes = new ShapeAnalysis(TENHOU_3P);
    shapes.reset();
    shapes.analyze(hands.move(discard, 0), tiles, ShapeAnalysis.UNSHARED);
    assertEquals(shapes.waits(), Tiles.bit(Tiles.S1));
    assertEquals(shapes.ukeire(), Tiles.bit(Tiles.S1));

    scene.step();
    scene.step();
    assertEquals(scene.legal(0), List.of("PON(1s)", "PASS"));
    at(scene, 0);
    assertEquals(r.tile(r.tokenizer.canonicalTile(Tiles.S1), TileField.CURRENT_WAIT), 1);
    assertEquals(r.tile(NORTH_TOKEN, TileField.CURRENT_WAIT), 0);
    assertEquals(r.tile(NORTH_TOKEN, TileField.CURRENT_UKEIRE), 0);
  }

  /** 三人打ちの牌の集合に無い種類は、正準の牌に写せない。 */
  @Test
  public void theMiddleManzuHasNoToken() {
    PlayedScene scene = PlayedScene.first().hand(0, NOTEN_A).hand(1, NOTEN_B).hand(2, NOTEN_C).draws("1z").start();
    at(scene, 0);
    expectThrows(IllegalStateException.class, () -> r.tokenizer.canonicalTile(Tiles.M5));
    assertFalse(scene.legal(0).isEmpty());
  }
}
