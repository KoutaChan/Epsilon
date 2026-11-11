package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 手番（ツモの後）と、チー・ポンの後の打牌の合法手を作る。合法手の並びは毎回同じになることが契約で、PRISM はこの並びを前提に
 * 共通乱数を使う。
 *
 * <p>立直していない手番の並び: ツモ和了 → 暗槓（種類の昇順）→ 加槓（面子の順）→ 北抜き → 立直の打牌 → 打牌 → 九種九牌。立直中は
 * ツモ和了 → 暗槓（ツモった種類で、待ちが変わらないとき）→ 北抜き（ツモった北だけ）→ ツモ切り。打牌と立直の打牌は種類の昇順で、
 * 同じ種類の中は、手の赤 → 手の赤でない牌 → ツモ切りの順。鳴いた後の打牌も同じ並びで、ツモ切りは無い。
 *
 * <p>槓できるのは、場の槓が 4 未満で、山の残りが 1 枚以上のとき。面子にできない種類（北を鳴けないルールの北）は槓できない。北抜きは、
 * 北を抜くルールで北を持ち、山の残りが 1 枚以上のとき。立直できるのは、門前（暗槓と北抜きは門前を崩さない）で、持ち点が 1000 点以上、
 * 山の残りが立直に要る枚数以上のとき。立直の打牌にできるのは、打った後の 13 枚に形の上の待ちが残る牌だけ。九種九牌は、鳴き・暗槓・
 * 加槓・北抜きの無い局で、自分の最初のツモのときに么九牌が 9 種以上あれば宣言できる。
 *
 * <p>鳴いた後の打牌、鳴いてできる面子、純手牌の赤 5 を求める公開メソッドは、席のビューだけを読む。そのため、エンジンの外（牌譜、
 * 1 席から見た卓、観測の記録を書く側）からも呼べる。
 */
public final class Legal {

  /** 九種九牌に要る么九牌の種類の数。 */
  private static final int NINE_KINDS = 9;

  private static final int TSUMO_CODE = Action.code(Type.TSUMO_AGARI);
  private static final int KYUSHU_CODE = Action.code(Type.KYUSHU_KYUHAI);
  private static final int NUKI_CODE = Action.code(Type.NUKI);

  private Legal() {}

  /**
   * 直前にツモった席の、手番の判断を作る。ツモ和了が合法なら、その点も判断に持たせる。
   *
   * @param wall 裏ドラ表示牌を読む牌山。立直の和了の点には裏ドラを含める。null なら裏ドラを数えない
   * @param id 次に使う判断 ID
   * @return 判断を作った後の、次に使う判断 ID
   */
  static long turn(PublicState state, Hand hand, Wall wall, Decisions decisions, long id, RuleWorkspace ws) {
    TurnEvent draw = state.last();
    if (draw.kind() != TurnEvent.Kind.DRAW) {
      throw new IllegalStateException("A turn follows a draw, not " + draw.kind());
    }
    int seat = draw.seat();
    int drawn = draw.tile();
    long reds = hand.redFives();
    Variant variant = state.variant();
    long meldable = variant.meldableKinds();
    boolean kans = state.kans() < Variant.MAX_KANS && state.liveLeft() > 0;
    boolean nuki = variant.has(RuleFlag.NUKI) && state.liveLeft() > 0 && hand.count(Tiles.PEI) > 0;
    decisions.clear();
    decisions.open(seat, EngineDecisionKind.TURN);
    if (WinLegality.tsumo(state, hand, wall, ws, ws.score())) {
      decisions.addWin(TSUMO_CODE, ws.score());
    }
    if (state.riichi(seat) != RiichiState.NONE) {
      if (kans
          && Tiles.contains(meldable, drawn)
          && hand.count(drawn) == Tiles.COPIES
          && keepsWaits(hand, drawn, ws)) {
        decisions.add(Action.code(Type.ANKAN, drawn, TileSelection.NOT_APPLICABLE));
      }
      if (nuki && drawn == Tiles.PEI) {
        decisions.add(NUKI_CODE);
      }
      decisions.add(Action.code(Type.DAHAI, drawn, TileSelection.discard(draw.red(), true)));
    } else {
      if (kans) {
        addKans(hand, meldable, decisions);
      }
      if (nuki) {
        decisions.add(NUKI_CODE);
      }
      if (mayRiichi(state, hand, seat)) {
        addRiichiDiscards(hand, drawn, draw.red(), reds, decisions, ws);
      }
      for (long kinds = hand.shape().kinds(); kinds != 0; kinds &= kinds - 1) {
        addDiscards(hand, Long.numberOfTrailingZeros(kinds), drawn, draw.red(), reds, Type.DAHAI, decisions);
      }
      if (nineTerminals(state, hand, seat)) {
        decisions.add(KYUSHU_CODE);
      }
    }
    return decisions.close(id) ? id + 1 : id;
  }

  /**
   * チー・ポンをした直後の席の、打牌の判断を作り、次に使う判断 ID を返す。鳴いた面子は手の最後の面子で、喰い替えになる種類は
   * 打てない。北抜きは合法手に入れない。
   */
  static long postCall(Hand hand, int seat, Decisions decisions, long id) {
    long allowed = hand.shape().kinds() & ~swapBan(Meld.of(hand.meld(hand.meldCount() - 1)));
    if (allowed == 0) {
      throw new IllegalStateException("No discard is left after the call: " + hand.shape());
    }
    long reds = hand.redFives();
    decisions.clear();
    decisions.open(seat, EngineDecisionKind.POST_CALL_DAHAI);
    for (long kinds = allowed; kinds != 0; kinds &= kinds - 1) {
      addDiscards(hand, Long.numberOfTrailingZeros(kinds), TurnEvent.NONE, false, reds, Type.DAHAI, decisions);
    }
    return decisions.close(id) ? id + 1 : id;
  }

  /**
   * 見る席が直前の他家の打牌をチーかポンで鳴いたとして、鳴いた後に打てる牌を行動の番号で書く。並びは鳴いた後の打牌の判断と同じ。
   * 手から出す牌は実際に鳴くときと同じで、赤を使う鳴きなら赤 5、ほかは赤でない牌を出す。喰い替えになる種類は除く。打牌は、残る
   * 純手牌の種類と赤の組み合わせごとに 1 つなので、書く先の長さは 11（配牌の 13 枚 − 2 枚）以上あれば足りる。鳴きの確かめと例外は
   * {@link #callMeld} と同じで、打てる牌が残らなければ IllegalStateException を投げる。
   *
   * @param view 鳴く席のビュー
   * @param callCode チー・ポンの行動の番号
   * @param out 打牌の行動の番号を先頭から書く先
   * @return 書いた数
   */
  public static int postCallDiscards(SeatView view, int callCode, short[] out) {
    Action call = Action.of(callCode);
    if (call.type() != Type.CHI && call.type() != Type.PON) {
      throw new IllegalArgumentException("Only a chi or a pon is followed by a discard: " + call);
    }
    Meld meld = called(view, call);
    long taken = takenKinds(call);
    int perKind = takenPerKind(call);
    long reds = redFives(view);
    if (call.selection().red()) {
      reds &= ~(taken & Tiles.FIVES);
    }
    long ban = swapBan(meld);
    int count = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int left = view.count(kind) - (Tiles.contains(taken, kind) ? perKind : 0);
      if (left == 0 || Tiles.contains(ban, kind)) {
        continue;
      }
      boolean red = Tiles.contains(reds, kind);
      if (red) {
        out[count++] = (short) Action.code(Type.DAHAI, kind, TileSelection.AKA);
      }
      if (left > (red ? 1 : 0)) {
        out[count++] = (short) Action.code(Type.DAHAI, kind, TileSelection.NON_AKA);
      }
    }
    if (count == 0) {
      throw new IllegalStateException("No discard is left after " + call + " by seat " + view.seat());
    }
    return count;
  }

  /**
   * 見る席が直前の他家の打牌を鳴いたとして、できる面子の正準の番号を返す。番号は赤の出どころも含む。赤の出どころは、エンジンが
   * 実際に鳴くときと同じ。鳴いた牌が赤なら鳴いた牌になる。そうでなければ、チー・ポンは赤を使う鳴きのとき、大明槓は手に赤 5 が
   * あるときに、手の牌が赤の出どころになる。
   *
   * <p>直前のイベントが他家の打牌でなければ IllegalStateException を投げる。次のときは IllegalArgumentException を投げる: 鳴きで
   * ない行動、ルールに無い鳴き、打牌と種類の合わない行動、上家以外からのチー、面子にできない種類のポン・大明槓、手の牌が足りない鳴き
   * （赤を使う鳴きで手に赤 5 が無いときと、赤を使わない鳴きで赤でない牌が足りないときを含む）。
   *
   * @param view 鳴く席のビュー
   * @param callCode チー・ポン・大明槓の行動の番号
   * @return 面子の正準の番号
   */
  public static int callMeld(SeatView view, int callCode) {
    return called(view, Action.of(callCode)).id();
  }

  /**
   * 鳴く席が直前の打牌を鳴いてできる面子（赤の出どころを含む）。鳴いた牌が赤なら、鳴いた牌が赤の出どころになる。そうでなければ、
   * チー・ポンは赤を使う鳴きのとき、大明槓は手に赤 5 があるときに、手の牌が赤の出どころになる。行動と手牌が鳴きに合うかは確かめない。
   *
   * @param call チー・ポン・大明槓
   * @param caller 鳴く席
   * @param discard 鳴かれる直前の打牌
   * @param handReds 鳴く席の純手牌に赤 5 がある種類のマスク
   * @param seats 席の数
   * @return 鳴いてできる面子
   */
  static Meld meldOf(Action call, int caller, TurnEvent discard, long handReds, int seats) {
    Meld.RelativeSource source = Meld.RelativeSource.of(caller, discard.seat(), seats);
    int kind = discard.tile();
    return switch (call.type()) {
      case PON -> Meld.pon(kind, source, aka(discard.red(), call.selection().red()));
      case DAIMINKAN -> Meld.daiminkan(kind, source, aka(discard.red(), Tiles.contains(handReds, kind)));
      case CHI -> Meld.chi(call.base(), kind, aka(discard.red(), call.selection().red()));
      default -> throw new IllegalArgumentException("Not a call: " + call);
    };
  }

  /** 見る席の純手牌に赤 5 がある種類のマスク。 */
  public static long redFives(SeatView view) {
    long reds = 0;
    for (long kinds = view.variant().tiles().redKinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      if (view.holds(Tiles.physicalId(kind, Tiles.RED_COPY))) {
        reds |= Tiles.bit(kind);
      }
    }
    return reds;
  }

  /**
   * チー・ポンの直後に、喰い替えになるので打てない種類。ポンは鳴いた種類。チーは鳴いた種類に加えて、鳴いた牌が順子の端なら、
   * 反対の端の 1 つ外にある同じ色の種類（筋）。面子の種別（チー・ポン）、順子、鳴いた種類だけで決まり、誰から鳴いたかと赤の出どころは問わない。
   */
  static long swapBan(Meld call) {
    return switch (call.type()) {
      case PON -> Tiles.bit(call.called());
      case CHI -> Tiles.bit(call.called()) | suji(call);
      default -> throw new IllegalArgumentException("Only a chi or a pon bans swaps: " + call);
    };
  }

  private static long suji(Meld chi) {
    int low = chi.base();
    int high = low + Meld.Type.CHI.tiles() - 1;
    Tiles.Group suit = Tiles.Group.of(low);
    if (chi.called() == low && high < suit.last()) {
      return Tiles.bit(high + 1);
    }
    if (chi.called() == high && low > suit.first()) {
      return Tiles.bit(low - 1);
    }
    return 0;
  }

  /** 見る席が直前の他家の打牌を鳴いてできる面子。打牌・行動・手の牌は、{@link #callMeld} に書いたとおりに確かめる。 */
  private static Meld called(SeatView view, Action call) {
    TurnEvent discard = view.last();
    int seat = view.seat();
    if (discard.kind() != TurnEvent.Kind.DISCARD || discard.seat() == seat) {
      throw new IllegalStateException("Seat " + seat + " can call only another seat's discard, not " + discard.kind());
    }
    Variant variant = view.variant();
    Type type = call.type();
    if (type != Type.CHI && type != Type.PON && type != Type.DAIMINKAN || !type.in(variant)) {
      throw new IllegalArgumentException(variant + " has no call " + call);
    }
    int called = discard.tile();
    if (call.tile() != called) {
      throw new IllegalArgumentException(call + " does not call the discarded kind " + called);
    }
    boolean fromLeft = Meld.RelativeSource.of(seat, discard.seat(), variant.seats()) == Meld.RelativeSource.KAMICHA;
    if (type == Type.CHI ? !fromLeft : !Tiles.contains(variant.meldableKinds(), called)) {
      throw new IllegalArgumentException(
          call + " is not open to seat " + seat + " on a discard of seat " + discard.seat());
    }
    long reds = redFives(view);
    int perKind = takenPerKind(call);
    for (long kinds = takenKinds(call); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      boolean red = Tiles.contains(reds, kind);
      boolean useRed = call.selection().red() && Tiles.contains(Tiles.FIVES, kind);
      if (useRed && !red) {
        throw new IllegalArgumentException(call + " needs a red five in the hand of seat " + seat);
      }
      // 大明槓は手のその種類を全て使う。チー・ポンは、赤を使う鳴きなら赤 1 枚と赤でない牌を、使わない鳴きなら赤でない牌だけを使う。
      int usable = type == Type.DAIMINKAN || useRed ? view.count(kind) : view.count(kind) - (red ? 1 : 0);
      if (usable < perKind) {
        throw new IllegalArgumentException(call + " needs more of kind " + kind + " in the hand of seat " + seat);
      }
    }
    return meldOf(call, seat, discard, reds, variant.seats());
  }

  /** 鳴きで手から面子へ出す種類。ポン・大明槓は鳴いた種類、チーは順子のほかの 2 種類。 */
  private static long takenKinds(Action call) {
    return call.type() == Type.CHI
        ? Tiles.range(call.base(), call.base() + Meld.Type.CHI.tiles() - 1) & ~Tiles.bit(call.tile())
        : Tiles.bit(call.tile());
  }

  /** 鳴きで手から面子へ出す、1 種類あたりの枚数。 */
  private static int takenPerKind(Action call) {
    return switch (call.type()) {
      case CHI -> 1;
      case PON -> Meld.Type.PON.tiles() - 1;
      case DAIMINKAN -> Meld.Type.DAIMINKAN.tiles() - 1;
      default -> throw new IllegalArgumentException("Not a call: " + call);
    };
  }

  private static Meld.AkaSource aka(boolean calledRed, boolean handRed) {
    if (calledRed) {
      return Meld.AkaSource.CALLED_TILE;
    }
    return handRed ? Meld.AkaSource.CONSUMED_HAND_TILE : Meld.AkaSource.NONE;
  }

  /** 暗槓と加槓を足す。暗槓は、純手牌に 4 枚ある種類のうち面子にできるものを昇順に足す。加槓は、4 枚目を純手牌に持つポンを面子の順に足す。 */
  private static void addKans(Hand hand, long meldable, Decisions decisions) {
    for (long kinds = hand.shape().pairs() & meldable; kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      if (hand.count(kind) == Tiles.COPIES) {
        decisions.add(Action.code(Type.ANKAN, kind, TileSelection.NOT_APPLICABLE));
      }
    }
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = Meld.of(hand.meld(index));
      if (meld.type() == Meld.Type.PON && hand.count(meld.base()) > 0) {
        decisions.add(Action.code(Type.KAKAN, meld.base(), TileSelection.NOT_APPLICABLE));
      }
    }
  }

  /** 立直中に暗槓できるなら true。ツモの前の 13 枚と、その種類を暗槓にした後の手とで、形の上の待ちが同じときに限る。 */
  private static boolean keepsWaits(Hand hand, int kind, RuleWorkspace ws) {
    HandShape probe = ws.probe();
    probe.copyFrom(hand.shape());
    probe.remove(kind);
    long before = ws.analyzer().waits(probe);
    for (int taken = 1; taken < Tiles.COPIES; taken++) {
      probe.remove(kind);
    }
    probe.addMeld();
    return ws.analyzer().waits(probe) == before;
  }

  /** 立直の条件のうち、打つ牌に依らないもの（門前・持ち点・山の残り）を満たすなら true。 */
  private static boolean mayRiichi(PublicState state, Hand hand, int seat) {
    if (state.score(seat) < Variant.DEPOSIT || state.liveLeft() < state.variant().riichiMinWall()) {
      return false;
    }
    for (int index = 0; index < hand.meldCount(); index++) {
      if (!Meld.of(hand.meld(index)).type().concealed()) {
        return false;
      }
    }
    return true;
  }

  /** 立直の打牌を足す。足すのは、打った後の 13 枚に形の上の待ちが残る種類だけ。 */
  private static void addRiichiDiscards(
      Hand hand, int drawn, boolean drawnRed, long reds, Decisions decisions, RuleWorkspace ws) {
    HandShape probe = ws.probe();
    for (long kinds = hand.shape().kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      probe.copyFrom(hand.shape());
      probe.remove(kind);
      if (ws.analyzer().waits(probe) != 0) {
        addDiscards(hand, kind, drawn, drawnRed, reds, Type.RIICHI_DAHAI, decisions);
      }
    }
  }

  private static boolean nineTerminals(PublicState state, Hand hand, int seat) {
    EventLog events = state.events();
    return state.variant().aborts(AbortiveDraw.NINE_TERMINALS)
        && events.interruptions() == 0
        && events.riverSize(seat) == 0
        && Long.bitCount(hand.shape().kinds() & Tiles.TERMINALS_AND_HONORS) >= NINE_KINDS;
  }

  /**
   * その種類の打牌を、手の赤 → 手の赤でない牌 → ツモ切りの順に足す。
   *
   * @param kind 打つ種類
   * @param drawn ツモった種類。鳴いた後は {@link TurnEvent#NONE}
   * @param drawnRed ツモった牌が赤 5 なら true
   * @param reds 純手牌に赤 5 がある種類のマスク（ツモった牌を含む）
   * @param type 打牌か立直の打牌
   */
  private static void addDiscards(
      Hand hand, int kind, int drawn, boolean drawnRed, long reds, Type type, Decisions decisions) {
    boolean fromDraw = kind == drawn;
    boolean red = Tiles.contains(reds, kind);
    int plain = hand.count(kind) - (red ? 1 : 0) - (fromDraw && !drawnRed ? 1 : 0);
    if (red && !(fromDraw && drawnRed)) {
      decisions.add(Action.code(type, kind, TileSelection.AKA));
    }
    if (plain > 0) {
      decisions.add(Action.code(type, kind, TileSelection.NON_AKA));
    }
    if (fromDraw) {
      decisions.add(Action.code(type, kind, TileSelection.discard(drawnRed, true)));
    }
  }
}
