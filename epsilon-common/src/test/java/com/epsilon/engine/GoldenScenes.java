package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Predicate;

/**
 * {@link GoldenGames} の作った場面。配牌と牌山を牌の id まで決め、台本の打ち方で稀な道（包・槍槓・赤を足す加槓・槓ドラの複数と遅れての
 * 表示・ダブロン・海底・河底・嶺上・四家立直・流し満貫）を必ず通す。
 *
 * <p>台本の打ち方: 許した席だけが和了り、許した鳴き・槓だけをし、許した席だけが立直する。ほかはツモ切り、鳴いた後は捨てる牌の表（junk）から、
 * 応答は見送り。手は配牌のまま動かないので、牌山の並びだけで局の筋が決まる。各場面は局の結果の期待（{@link Scene#expect}）を持ち、筋が外れたら
 * 試験が落ちる（稀な道を本当に通ったことの保証）。
 *
 * <p>牌山: 位置 0..51 は配牌（席 s の 13 枚を 13s から）、52.. はツモの順、130−2i は i 枚目の表示牌、131−2i はその裏、135..132 は嶺上牌。
 * 指定のない位置は残りの牌を場面ごとの種で混ぜて埋める。{@code excluded} の牌の種類は埋め草の最後に回し、王牌（表示牌・裏・嶺上の空き）にだけ
 * 置く（和了の牌を早く引いて筋が外れないように）。
 */
final class GoldenScenes {
  private static final int DEAL = 52;
  private static final int LIVE_END = 122;
  private static final int WALL = Tile.NUM_TILES;

  private GoldenScenes() {}

  /** 和了を許す時。 */
  enum WinWhen {
    /** いつでも。 */
    ALWAYS,
    /** 槍槓の応答でだけ。 */
    CHANKAN,
    /** 牌山の最後の牌（海底のツモ・河底のロン）でだけ。 */
    LAST_TILE;

    boolean allows(EngineDecisionPoint point, GameState state) {
      return switch (this) {
        case ALWAYS -> true;
        case CHANKAN -> point.kind() == EngineDecisionKind.CHANKAN;
        case LAST_TILE -> state.remainingWallTiles() == 0;
      };
    }
  }

  /** 許した鳴き・槓（席・種類・牌の種類）。 */
  private record Call(int seat, Action.Type type, int tileType) {}

  /** 表記の 1 枚（牌の種類と、赤の 5 か）。 */
  record SceneTile(int type, boolean aka) {}

  /** 1 つの場面。組み立てた後は変えない。 */
  static final class Scene {
    final String name;
    private final int honba;
    private final int kyotaku;
    private final String[] hands = new String[GameState.NUM_PLAYERS];
    private final String[] wall = new String[WALL];
    private final long[] junk = new long[GameState.NUM_PLAYERS];
    private final List<Call> calls = new ArrayList<>();
    private long excluded;
    private int winners;
    private WinWhen winWhen = WinWhen.ALWAYS;
    private int riichi;
    private Predicate<RoundResult> expect = result -> true;

    private Scene(String name, int honba, int kyotaku) {
      this.name = name;
      this.honba = honba;
      this.kyotaku = kyotaku;
    }

    private Scene hand(int seat, String notation) {
      hands[seat] = notation;
      return this;
    }

    /** {@code k} 番目のツモ（位置 52+k）から順に置く。{@code *} は埋め草。 */
    private Scene draws(int k, String tiles) {
      String[] tokens = tiles.trim().split("\\s+");
      for (int i = 0; i < tokens.length; i++) {
        if (!tokens[i].equals("*")) wall[DEAL + k + i] = tokens[i];
      }
      return this;
    }

    /** 嶺上牌を引く順（位置 135, 134, ...）に置く。 */
    private Scene rinshan(String tiles) {
      String[] tokens = tiles.trim().split("\\s+");
      for (int i = 0; i < tokens.length; i++) wall[WALL - 1 - i] = tokens[i];
      return this;
    }

    private Scene junk(int seat, String notation) {
      for (SceneTile tile : parse(notation)) junk[seat] |= 1L << tile.type();
      return this;
    }

    private Scene call(int seat, Action.Type type, String tile) {
      calls.add(new Call(seat, type, parse(tile).getFirst().type()));
      return this;
    }

    private Scene exclude(String notation) {
      for (SceneTile tile : parse(notation)) excluded |= 1L << tile.type();
      return this;
    }

    private Scene winners(WinWhen when, int... seats) {
      winWhen = when;
      for (int seat : seats) winners |= 1 << seat;
      return this;
    }

    private Scene riichi(int... seats) {
      for (int seat : seats) riichi |= 1 << seat;
      return this;
    }

    private Scene expect(Predicate<RoundResult> expectation) {
      expect = expectation;
      return this;
    }

    /** 局の結果が場面の筋どおりか。 */
    boolean expected(RoundResult result) {
      return expect.test(result);
    }

    /** 場面の局を始めたエンジン（東 1 局、親は席 0、牌山と配牌は場面のもの）。 */
    GameEngine start() {
      GameEngine engine = new GameEngine(GoldenHash.mix(name.hashCode()));
      GameState state = engine.getState();
      state.startRound(0, 0, honba, kyotaku);
      int[] tiles = layout();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        Hand hand = state.hand(seat);
        hand.clear();
        for (int index = 0; index < 13; index++) hand.addPhysicalTile(tiles[seat * 13 + index]);
      }
      state.initializeWall(tiles);
      return engine;
    }

    /** 136 枚の物理牌の並び。指定の牌を先に取り、残りで埋める。 */
    private int[] layout() {
      int[] tiles = new int[WALL];
      boolean[] used = new boolean[WALL];
      boolean[] placed = new boolean[WALL];
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        if (hands[seat] == null) continue;
        List<SceneTile> hand = parse(hands[seat]);
        if (hand.size() != 13) throw new IllegalStateException(name + ": hand " + seat + " is not 13 tiles");
        for (int index = 0; index < 13; index++) {
          tiles[seat * 13 + index] = take(used, hand.get(index));
          placed[seat * 13 + index] = true;
        }
      }
      for (int position = 0; position < WALL; position++) {
        if (wall[position] == null) continue;
        tiles[position] = take(used, parse(wall[position]).getFirst());
        placed[position] = true;
      }
      int[] fillers = fillers(used);
      int next = 0;
      for (int position = 0; position < WALL; position++) {
        if (placed[position]) continue;
        int tile = fillers[next++];
        if (position < LIVE_END && (excluded >>> Tile.typeOf(tile) & 1) != 0) {
          throw new IllegalStateException(name + ": excluded tile reached position " + position);
        }
        tiles[position] = tile;
      }
      return tiles;
    }

    /** 残りの牌を場面の種で混ぜ、除いた種類を最後に回した並び。 */
    private int[] fillers(boolean[] used) {
      int[] free = new int[WALL];
      int count = 0;
      for (int tile = 0; tile < WALL; tile++) if (!used[tile]) free[count++] = tile;
      SplittableRandom random = new SplittableRandom(name.hashCode());
      for (int index = count - 1; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int swap = free[index];
        free[index] = free[other];
        free[other] = swap;
      }
      int[] ordered = new int[count];
      int front = 0;
      for (int index = 0; index < count; index++) {
        if ((excluded >>> Tile.typeOf(free[index]) & 1) == 0) ordered[front++] = free[index];
      }
      for (int index = 0; index < count; index++) {
        if ((excluded >>> Tile.typeOf(free[index]) & 1) != 0) ordered[front++] = free[index];
      }
      return ordered;
    }

    /** 台本の打ち方で 1 つの判断の手を選ぶ。 */
    Action choose(GameEngine engine, EngineDecisionPoint point) {
      List<Action> legal = point.legalActions();
      int seat = point.player();
      if ((winners >>> seat & 1) != 0 && winWhen.allows(point, engine.getState())) {
        Action win = first(legal, Action.Type.TSUMO_AGARI, -1);
        if (win == null) win = first(legal, Action.Type.RON_AGARI, -1);
        if (win != null) return win;
      }
      for (Call call : calls) {
        if (call.seat() != seat) continue;
        Action action = first(legal, call.type(), call.tileType());
        if (action != null) return action;
      }
      if ((riichi >>> seat & 1) != 0) {
        Action declaration = tsumogiri(legal, Action.Type.RIICHI_DAHAI);
        if (declaration == null) declaration = first(legal, Action.Type.RIICHI_DAHAI, -1);
        if (declaration != null) return declaration;
      }
      Action discard = tsumogiri(legal, Action.Type.DAHAI);
      if (discard != null) return discard;
      for (Action action : legal) {
        if (action.type() == Action.Type.DAHAI && (junk[seat] >>> action.tileType() & 1) != 0) {
          return action;
        }
      }
      Action pass = first(legal, Action.Type.PASS, -1);
      return pass != null ? pass : legal.getFirst();
    }
  }

  private static Action first(List<Action> legal, Action.Type type, int tileType) {
    for (Action action : legal) {
      if (action.type() == type && (tileType < 0 || action.tileType() == tileType)) return action;
    }
    return null;
  }

  private static Action tsumogiri(List<Action> legal, Action.Type type) {
    for (Action action : legal) {
      if (action.type() == type && action.isTsumogiri()) return action;
    }
    return null;
  }

  /** 牌の表記（例 {@code 556677z123m9s}、{@code 0p} は赤の 5p）を、表記の順の牌の列にする。空白は読み飛ばす。 */
  static List<SceneTile> parse(String notation) {
    List<SceneTile> tiles = new ArrayList<>();
    List<Integer> digits = new ArrayList<>();
    for (char c : notation.toCharArray()) {
      if (Character.isDigit(c)) {
        digits.add(c - '0');
        continue;
      }
      int offset =
          switch (c) {
            case 'm' -> Tile.M1;
            case 'p' -> Tile.P1;
            case 's' -> Tile.S1;
            case 'z' -> Tile.TON;
            case ' ' -> -1;
            default -> throw new IllegalArgumentException("bad tile notation: " + notation);
          };
      if (offset < 0) continue;
      for (int digit : digits) {
        boolean aka = digit == 0;
        tiles.add(new SceneTile(offset + (aka ? 5 : digit) - 1, aka));
      }
      digits.clear();
    }
    return tiles;
  }

  /** 指定の牌の物理 id を取る。赤は 5 の 0 枚目、赤でない 5 は 1..3 枚目、ほかは 0..3 枚目の空きの最初。 */
  private static int take(boolean[] used, SceneTile tile) {
    int type = tile.type();
    boolean aka = tile.aka();
    for (int copy = 0; copy < Tile.TILES_PER_TYPE; copy++) {
      int id = type * Tile.TILES_PER_TYPE + copy;
      if (used[id] || Tile.isAka(id) != aka) continue;
      used[id] = true;
      return id;
    }
    throw new IllegalStateException("no copy left of tile type " + type + (aka ? " (aka)" : ""));
  }

  /** 場面の一覧（試験の順）。 */
  static List<Scene> all() {
    List<Scene> scenes = new ArrayList<>();
    // 大三元の包: 席 1 が 3 枚目の三元牌を席 2 から鳴き、席 3 からロン（包と放銃者で折半）・ツモ（包が全額）
    scenes.add(daisangen("pao-daisangen-ron", "5z 6z 7z 2m 9s").expect(result -> paoRon(result, 3)));
    scenes.add(daisangen("pao-daisangen-tsumo", "5z 6z 7z 2m 3m 4m 9s").expect(GoldenScenes::paoTsumo));
    // 大四喜の包: 4 枚目の風を席 2 から鳴く
    scenes.add(daisuushii("pao-daisuushii-ron", "1z 2z 3z 4z 2m 5m").expect(result -> paoRon(result, 3)));
    scenes.add(daisuushii("pao-daisuushii-tsumo", "1z 2z 3z 4z 2m 3m 4m 5m").expect(GoldenScenes::paoTsumo));
    // 槍槓: 席 1 が 3m をポンして 4 枚目で加槓、席 2 の嵌張待ち（役は槍槓だけ）がロン
    scenes.add(
        new Scene("chankan", 0, 0)
            .hand(1, "33m456p789p123s9m1z")
            .hand(2, "12m456s789s234p99p")
            .junk(1, "1z")
            .call(1, Action.Type.PON, "3m")
            .call(1, Action.Type.KAKAN, "3m")
            .winners(WinWhen.CHANKAN, 2)
            .draws(0, "3m * * * 3m")
            .expect(result -> result instanceof RoundResult.RonAgari ron && ron.loser() == 1 && ron.hasWinner(2)));
    // 槓 3 つ: ポンの後の暗槓（すぐ表示）、赤の 5p を足す加槓（打牌の後に表示）、大明槓の嶺上ツモ（表示を待つまま和了）
    scenes.add(
        new Scene("kans-rinshan", 0, 0)
            .hand(1, "9999p55p777s234m1z")
            .junk(1, "1z")
            .call(1, Action.Type.PON, "5p")
            .call(1, Action.Type.ANKAN, "9p")
            .call(1, Action.Type.KAKAN, "5p")
            .call(1, Action.Type.DAIMINKAN, "7s")
            .winners(WinWhen.ALWAYS, 1)
            .draws(0, "5p * * * 8m * * * 0p 7s")
            .rinshan("2z 3z 8m")
            .exclude("8m")
            .expect(result -> result instanceof RoundResult.TsumoAgari tsumo && tsumo.winner() == 1));
    // 海底: 席 1 が最後の牌（位置 121）でツモ。河底: 席 1 の最後の打牌を席 2 がロン
    scenes.add(
        new Scene("haitei", 0, 0)
            .hand(1, "123m456p789s11z23s")
            .winners(WinWhen.LAST_TILE, 1)
            .draws(69, "1s")
            .exclude("1s4s")
            .expect(result -> result instanceof RoundResult.TsumoAgari tsumo && tsumo.winner() == 1));
    scenes.add(
        new Scene("houtei", 0, 0)
            .hand(2, "123m456p789s11z23s")
            .winners(WinWhen.LAST_TILE, 2)
            .draws(69, "4s")
            .exclude("1s4s")
            .expect(result -> result instanceof RoundResult.RonAgari ron && ron.loser() == 1 && ron.hasWinner(2)));
    // 流し満貫: 席 3 のツモ（位置 55 + 4j）を全て么九牌にする。最初の打牌は風でない（四風連打にしない）
    scenes.add(
        new Scene("nagashi", 0, 0)
            .draws(3, nagashiDraws())
            .expect(
                result -> result instanceof RoundResult.ExhaustiveDraw draw && draw.isNagashiWinner(3)));
    // 四家立直: 全員が門前聴牌の配牌で、最初の打牌で立直する（ダブル立直）
    scenes.add(
        new Scene("four-riichi", 0, 0)
            .hand(0, "123m456p789s11z23s")
            .hand(1, "456m789p123s22z67p")
            .hand(2, "789m123p456s33z23p")
            .hand(3, "345m345p345s44z67m")
            .riichi(0, 1, 2, 3)
            .expect(
                result ->
                    result instanceof RoundResult.AbortiveDraw abort
                        && abort.reason() == RoundResult.AbortiveDrawReason.FOUR_RIICHI));
    // ダブロン（2 本場・供託 1 本）: 親の最初の打牌 6s に席 1 と席 3 が役牌の刻子でロン
    scenes.add(
        new Scene("double-ron", 2, 1)
            .hand(1, "555z234m678p99s45s")
            .hand(3, "666z345m456p11s45s")
            .winners(WinWhen.ALWAYS, 1, 3)
            .draws(0, "6s")
            .expect(
                result ->
                    result instanceof RoundResult.RonAgari ron
                        && ron.claims().size() == 2
                        && ron.loser() == 0));
    return scenes;
  }

  /** 席 1 が三元牌を対子で 3 組持ち、鳴いた後に 1p・9p・1s を捨て、9s 単騎になる。 */
  private static Scene daisangen(String name, String draws) {
    return new Scene(name, 1, 1)
        .hand(1, "556677z123m9s19p1s")
        .junk(1, "19p1s")
        .call(1, Action.Type.PON, "5z")
        .call(1, Action.Type.PON, "6z")
        .call(1, Action.Type.PON, "7z")
        .winners(WinWhen.ALWAYS, 1)
        .draws(0, draws);
  }

  /** 席 1 が風牌を対子で 4 組持ち、鳴いた後に 1p・9p・1s・9s を捨て、5m 単騎になる。 */
  private static Scene daisuushii(String name, String draws) {
    return new Scene(name, 1, 1)
        .hand(1, "11223344z5m19p19s")
        .junk(1, "19p19s")
        .call(1, Action.Type.PON, "1z")
        .call(1, Action.Type.PON, "2z")
        .call(1, Action.Type.PON, "3z")
        .call(1, Action.Type.PON, "4z")
        .winners(WinWhen.ALWAYS, 1)
        .draws(0, draws);
  }

  /** 席 3 の 17 回のツモ（k = 3 + 4j）を么九牌に、間の 3 回は埋め草にする。 */
  private static String nagashiDraws() {
    String[] terminals = "1m 9m 1p 9p 1s 9s 1z 2z 3z 4z 5z 6z 7z 1m 9m 1p 9p".split(" ");
    StringBuilder draws = new StringBuilder();
    for (int j = 0; j < terminals.length; j++) {
      if (j > 0) draws.append(" * * * ");
      draws.append(terminals[j]);
    }
    return draws.toString();
  }

  /** 席 1 が席 {@code loser} からロンし、包の席 2 も払う。 */
  private static boolean paoRon(RoundResult result, int loser) {
    return result instanceof RoundResult.RonAgari ron
        && ron.loser() == loser
        && ron.hasWinner(1)
        && ron.pointDelta().get(2) < 0;
  }

  /** 席 1 のツモを包の席 2 が全て払う。 */
  private static boolean paoTsumo(RoundResult result) {
    PointDelta delta = result.pointDelta();
    return result instanceof RoundResult.TsumoAgari tsumo
        && tsumo.winner() == 1
        && delta.get(2) < 0
        && delta.get(0) == 0
        && delta.get(3) == 0;
  }
}
