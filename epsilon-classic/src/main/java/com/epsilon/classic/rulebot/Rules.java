package com.epsilon.classic.rulebot;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;

/**
 * ルールベースの打ち手の判断の手順と、1 スレッド分の作業領域。手順は {@link RuleBot} の説明のとおり。判断のたびに作業領域を使い回し、
 * メモリは確保しない。
 *
 * <p>聴牌の手の攻めの点は、役のある待ちの枚数 × 10000 ＋ ロンの枚数 × 2500 ＋ ロンの平均の点数 × ロンの枚数 ÷ 4 ＋ ドラと赤の翻 × 1600。
 * 振聴になる打牌では、ロンの枚数を 0 とする。聴牌でない手は、受け入れ × 1000 ＋ 2 枚目の受け入れ × 160 ＋ 受け入れの種類 × 80 ＋
 * ドラと赤の翻 × 1600 ＋ 面子で決まった役 × 400。押すときは、危険度 1 につき 300 を引く。
 */
final class Rules {

  /** 押し引きの方針。 */
  private enum Mode {
    /** 脅威が無く、攻めの点だけで選ぶ。 */
    NORMAL,
    /** 脅威があっても押し、危険度の分を攻めの点から引く。 */
    PRESS,
    /** 降りる。危険度の小さい牌を優先して選ぶ。 */
    FOLD
  }

  private static final long UKEIRE_WEIGHT = 1_000;
  private static final long SECOND_ORDER_WEIGHT = 160;
  private static final long UKEIRE_KIND_WEIGHT = 80;
  private static final long VALUE_HAN_WEIGHT = 1_600;
  private static final long CONFIRMED_YAKU_WEIGHT = 400;
  private static final long LIVE_WAIT_WEIGHT = 10_000;
  private static final long RON_WAIT_WEIGHT = 2_500;
  private static final long RON_POINT_DIVISOR = 4;
  private static final long PRESS_DANGER_WEIGHT = 300;

  /** 打点があると見る、ロンのいちばん安い点数の下限。 */
  private static final int VALUABLE_RON = 5_200;

  /** ダマで十分と見る、ロンのいちばん安い点数の下限。 */
  private static final int DAMATEN_RON = 7_700;

  /** 大明槓をする条件。役のある待ちの生きている枚数の下限と、ロンのいちばん安い点数の範囲。 */
  private static final int KAN_LIVE_WAITS = 5;

  private static final int KAN_MIN_RON = 2_000;
  private static final int KAN_MAX_RON = 5_200;

  /** 1 向聴で押すのに要る受け入れの枚数・種類の下限と、押し引きで見る自分のツモの残りの回数の下限。 */
  private static final int WIDE_UKEIRE = 8;

  private static final int WIDE_UKEIRE_KINDS = 3;
  private static final int EARLY_SELF_DRAWS = 5;

  /** 押すとき、宣言牌の危険度がこれを超えたら立直を控える。 */
  private static final int RIICHI_PRESS_DANGER = 50;

  /** 待ちが悪くても立直する巡目（この巡目以降）。 */
  private static final int LATE_RIICHI_TURN = 8;

  /** 九種九牌で流さずに続ける向聴の上限。 */
  private static final int KYUSHU_CONTINUE_SHANTEN = 2;

  /** 槓の後に残っていなければならない受け入れの割合（百分率）。 */
  private static final int KAN_UKEIRE_PERCENT = 80;

  private static final int PERCENT = 100;

  /** 鳴くときの、鳴いた後の向聴の上限（ドラと赤が 2 翻未満で、役牌のポンでもないとき）。 */
  private static final int CALL_MAX_SHANTEN = 2;

  /** 1 つの判断で使う解析と候補の数の上限。応答の時点の手に 1 つと、合法手ごと・鳴いた後の打牌ごとに 1 つずつ。 */
  private static final int CAPACITY = 1 + Decisions.MAX_LEGAL + ObservationSchema.MAX_TRANSITIONS;

  private final Situation situation;
  private final Projection projection;
  private final Quality[] qualities = new Quality[CAPACITY];
  private final Candidate[] candidates = new Candidate[CAPACITY];

  /** 合法手ごとの打牌の解析。結果が同じ打牌が複数あるときは、代表でない手も代表の解析を指す。 */
  private final Quality[] discards = new Quality[Decisions.MAX_LEGAL];

  private int usedQualities;
  private int usedCandidates;
  private Observation o;

  Rules(Variant variant) {
    situation = new Situation(variant);
    projection = new Projection(variant);
    for (int index = 0; index < CAPACITY; index++) {
      qualities[index] = new Quality();
      candidates[index] = new Candidate();
    }
  }

  /** 判断で行う手を選び、合法手の何番目かを返す。 */
  int choose(Observation o) {
    this.o = o;
    int win = find(Action.Type.TSUMO_AGARI);
    if (win < 0) {
      win = find(Action.Type.RON_AGARI);
    }
    if (win >= 0) {
      return win;
    }
    int nuki = find(Action.Type.NUKI);
    if (nuki >= 0) {
      return nuki;
    }
    situation.load(o);
    projection.load(o, situation);
    usedQualities = 0;
    usedCandidates = 0;
    int pass = find(Action.Type.PASS);
    return pass >= 0 ? respond(pass) : turn();
  }

  /** 他家の打牌に鳴くか見送るかを選ぶ。脅威があれば見送る。脅威が無ければ、鳴く条件を満たすうちでいちばん良い鳴きを選び、無ければ見送る。 */
  private int respond(int pass) {
    if (situation.threats() != 0) {
      return pass;
    }
    Quality current = projection.current(nextQuality());
    Candidate best = null;
    int callDiscard = 0;
    for (int action = 0; action < o.legalActions(); action++) {
      int count = o.callDiscardsOf(action);
      Candidate choice =
          switch (o.action(action).type()) {
            case CHI, PON -> callAndDiscard(action, callDiscard, count);
            case DAIMINKAN -> callKan(action);
            default -> null;
          };
      callDiscard += count;
      if (choice != null && accept(current, choice) && (best == null || betterCall(choice, best))) {
        best = choice;
      }
    }
    return best == null ? pass : best.index;
  }

  /** チー・ポンを、鳴いた後にいちばん良い打牌（押し引きは考えない）をした手で評価する。 */
  private Candidate callAndDiscard(int action, int first, int count) {
    Action call = o.action(action);
    Candidate best = null;
    for (int transition = 0; transition < count; transition++) {
      Action discard = o.callDiscard(first + transition);
      Quality quality = projection.call(call, discard, nextQuality());
      Candidate candidate =
          nextCandidate().set(action, discard, quality, situation.danger(discard.tile()), offense(quality, false));
      if (best == null || betterDiscard(candidate, best, Mode.NORMAL)) {
        best = candidate;
      }
    }
    return best == null ? null : nextCandidate().set(action, call, best.quality, 0, best.score);
  }

  private Candidate callKan(int action) {
    Action kan = o.action(action);
    Quality quality = projection.kan(kan, RiichiState.NONE, nextQuality());
    return nextCandidate().set(action, kan, quality, 0, offense(quality, false));
  }

  /**
   * 鳴くかを決める。鳴いた後に役が見込めることが前提。大明槓は、槓の後が役のある待ち 5 枚以上の聴牌で、ロンのいちばん安い点数が
   * 2000〜5200 点のときだけする。チー・ポンは、向聴が進み、さらに 2 向聴以内になるか、ドラと赤が 2 翻以上か、役牌のポンのときにする。
   * 向聴が変わらないときは、1 向聴以内の役牌のポンで、攻めの点が上がるときだけする。
   */
  private boolean accept(Quality current, Candidate choice) {
    Quality after = choice.quality;
    if (!reliableOpenYaku(after)) {
      return false;
    }
    if (choice.action.type() == Action.Type.DAIMINKAN) {
      return after.shanten() == 0
          && after.liveWaits() >= KAN_LIVE_WAITS
          && after.minimumRon() >= KAN_MIN_RON
          && after.minimumRon() <= KAN_MAX_RON;
    }
    boolean yakuhaiPon = choice.action.type() == Action.Type.PON && situation.yakuhai(choice.action.tile());
    if (after.shanten() > current.shanten()) {
      return false;
    }
    if (after.shanten() == current.shanten()) {
      return yakuhaiPon && after.shanten() <= 1 && choice.score > offense(current, false);
    }
    return after.shanten() <= CALL_MAX_SHANTEN || after.valueHan() >= 2 || yakuhaiPon;
  }

  /** 自分の手番の手を選ぶ。九種九牌、打牌（押し引き）、立直、槓の順に決める。 */
  private int turn() {
    int abort = find(Action.Type.KYUSHU_KYUHAI);
    if (abort >= 0 && !situation.lateLast() && projection.far(KYUSHU_CONTINUE_SHANTEN)) {
      return abort;
    }
    int damaFrom = usedCandidates;
    evaluateDiscards(Action.Type.DAHAI, situation.riichi());
    int riichiFrom = usedCandidates;
    evaluateDiscards(Action.Type.RIICHI_DAHAI, situation.declaration());
    int riichiTo = usedCandidates;
    Candidate dama = best(damaFrom, riichiFrom, Mode.NORMAL);
    Candidate riichi = best(riichiFrom, riichiTo, Mode.NORMAL);
    if (dama == null && riichi == null) {
      return 0;
    }
    Mode mode = mode(dama != null ? dama : riichi);
    Candidate selected = dama != null && mode != Mode.NORMAL ? best(damaFrom, riichiFrom, mode) : dama;
    if (selected == null) {
      selected = riichi;
    }
    if (riichi != null && declareRiichi(riichi, dama, mode)) {
      selected = riichi;
    }
    int kan = kan(selected, mode);
    return kan >= 0 ? kan : selected.index;
  }

  /**
   * 1 つの種別の打牌を全て解析して、候補に並べる。結果が同じ打牌が複数あるときは、代表でない手にも代表の解析を使う。
   *
   * @param type 打牌か、立直を宣言する打牌
   * @param riichi 打牌の後の立直の状態
   */
  private void evaluateDiscards(Action.Type type, RiichiState riichi) {
    for (int action = 0; action < o.legalActions(); action++) {
      Action discard = o.action(action);
      if (discard.type() != type) {
        continue;
      }
      int twin = o.representative(action);
      Quality quality = twin == action ? projection.discard(discard, riichi, nextQuality()) : discards[twin];
      discards[action] = quality;
      boolean furiten = situation.furiten(quality.waits(), discard.tile());
      nextCandidate().set(action, discard, quality, situation.danger(discard.tile()), offense(quality, furiten));
    }
  }

  /**
   * 並べた候補のうち指定の範囲から、その押し引きでいちばん良い打牌を選ぶ。
   *
   * @param from 範囲の最初の候補の番号
   * @param to 範囲の最後の候補の次の番号
   * @return 最良の候補（範囲が空なら null）
   */
  private Candidate best(int from, int to, Mode mode) {
    Candidate best = null;
    for (int index = from; index < to; index++) {
      if (best == null || betterDiscard(candidates[index], best, mode)) {
        best = candidates[index];
      }
    }
    return best;
  }

  /**
   * 押し引きを決める。脅威が無ければ普通に打つ。終盤の最下位なら押す。終盤の 1 位は、聴牌でなければ降りる。それ以外は、次の 4 つの
   * うちいくつ当てはまるかで決める: 良い待ち、打点（ドラと赤が 2 翻以上か、ロンのいちばん安い点数が 5200 点以上）、親、自分のツモが
   * 5 回以上残る。聴牌なら 2 つ以上（立直した他家が 2 人以上なら 3 つ以上）で押す。1 向聴は、立直した他家が 1 人以下で、受け入れが広く
   * （8 枚以上・3 種類以上）、ツモが 5 回以上残り、打点があるか親のときだけ押す。2 向聴以上は降りる。
   */
  private Mode mode(Candidate attack) {
    if (situation.threats() == 0) {
      return Mode.NORMAL;
    }
    if (situation.lateLast()) {
      return Mode.PRESS;
    }
    Quality quality = attack.quality;
    if (situation.lateFirst() && quality.shanten() > 0) {
      return Mode.FOLD;
    }
    boolean valuable = quality.valueHan() >= 2 || quality.minimumRon() >= VALUABLE_RON;
    boolean early = situation.selfDrawsLeft() >= EARLY_SELF_DRAWS;
    int positive = (quality.goodWait() ? 1 : 0) + (valuable ? 1 : 0) + (situation.dealer() ? 1 : 0) + (early ? 1 : 0);
    if (quality.shanten() == 0) {
      return positive >= (situation.riichiThreats() >= 2 ? 3 : 2) ? Mode.PRESS : Mode.FOLD;
    }
    if (quality.shanten() == 1) {
      boolean wide = quality.ukeire() >= WIDE_UKEIRE && quality.ukeireKinds() >= WIDE_UKEIRE_KINDS;
      boolean press = situation.riichiThreats() <= 1 && wide && early && (valuable || situation.dealer());
      return press ? Mode.PRESS : Mode.FOLD;
    }
    return Mode.FOLD;
  }

  /**
   * 立直するかを決める。前提は、まだ立直しておらず、立直した後も役のある待ちが残り、降りではないこと。押すときは、宣言牌の危険度が
   * 50 以下であることも前提にする（終盤の最下位なら問わない）。ダマでロンできる待ちが無ければ立直する。終盤の 1 位のときと、ダマの
   * ロンが最低でも 7700 点のときはダマにする。ほかは、良い待ちか、待ちが 3 枚以上か、親か、8 巡目以降なら立直する。
   */
  private boolean declareRiichi(Candidate riichi, Candidate dama, Mode mode) {
    if (situation.riichi() != RiichiState.NONE || riichi.quality.liveWaits() == 0 || mode == Mode.FOLD) {
      return false;
    }
    if (mode == Mode.PRESS && riichi.danger > RIICHI_PRESS_DANGER && !situation.lateLast()) {
      return false;
    }
    if (dama == null || dama.quality.liveRon() == 0) {
      return true;
    }
    if (situation.lateFirst() || dama.quality.minimumRon() >= DAMATEN_RON) {
      return false;
    }
    return riichi.quality.goodWait()
        || riichi.quality.liveWaits() >= 3
        || situation.dealer()
        || situation.turn() >= LATE_RIICHI_TURN;
  }

  /**
   * 暗槓・加槓をするかを決める。槓をするのは、脅威が無く普通に打つときで、七対子の 1 向聴以内でないときだけ。候補にするのは、槓の後が
   * 1 向聴以内で、選んだ打牌より向聴が悪くならない槓。向聴が同じなら、受け入れが 8 割以上残ることも要る。加槓は、槓の後が良い待ちの
   * 聴牌で、ドラと赤が 2 翻以上のときだけ候補にする。候補のうち、いちばん良いものを選ぶ。
   *
   * @return 槓の合法手の何番目か（しなければ −1）
   */
  private int kan(Candidate selected, Mode mode) {
    if (mode != Mode.NORMAL || situation.threats() != 0) {
      return -1;
    }
    Quality chosen = selected.quality;
    if (chosen.special() && chosen.chiitoitsuShanten() <= 1) {
      return -1;
    }
    int best = -1;
    Quality bestQuality = null;
    for (int action = 0; action < o.legalActions(); action++) {
      Action kan = o.action(action);
      if (kan.type() != Action.Type.ANKAN && kan.type() != Action.Type.KAKAN) {
        continue;
      }
      Quality after = projection.kan(kan, situation.riichi(), nextQuality());
      if (after.shanten() > 1 || after.shanten() > chosen.shanten()) {
        continue;
      }
      if (after.shanten() == chosen.shanten() && after.ukeire() * PERCENT < chosen.ukeire() * KAN_UKEIRE_PERCENT) {
        continue;
      }
      if (kan.type() == Action.Type.KAKAN && (after.valueHan() < 2 || after.shanten() > 0 || !after.goodWait())) {
        continue;
      }
      if (bestQuality == null || betterQuality(after, bestQuality)) {
        best = action;
        bestQuality = after;
      }
    }
    return best;
  }

  /** 攻めの点を返す（式はクラスの説明のとおり）。 */
  private static long offense(Quality quality, boolean furiten) {
    if (quality.shanten() == 0) {
      int ron = furiten ? 0 : quality.liveRon();
      return quality.liveWaits() * LIVE_WAIT_WEIGHT
          + ron * RON_WAIT_WEIGHT
          + (long) quality.averageRon() * ron / RON_POINT_DIVISOR
          + quality.valueHan() * VALUE_HAN_WEIGHT;
    }
    return quality.ukeire() * UKEIRE_WEIGHT
        + quality.secondOrder() * SECOND_ORDER_WEIGHT
        + quality.ukeireKinds() * UKEIRE_KIND_WEIGHT
        + quality.valueHan() * VALUE_HAN_WEIGHT
        + quality.confirmedYaku() * CONFIRMED_YAKU_WEIGHT;
  }

  /** 鳴いた後に役が見込めるか。聴牌なら役のある待ちが残ること。聴牌でなければ、面子で決まった役か、断么九か、1 色と字牌だけの手。 */
  private static boolean reliableOpenYaku(Quality quality) {
    if (quality.shanten() == 0) {
      return quality.liveWaits() > 0;
    }
    return quality.confirmedYaku() != 0 || quality.allSimple() || quality.oneSuit();
  }

  /**
   * 打牌を比べ、候補のほうが良ければ true を返す。比べる順は、降りなら危険度、次に向聴、押し引きでの点、残る赤の数、危険度、赤を
   * 打たないほう、ツモ切りのほう、牌の種類の小さいほう。
   */
  private static boolean betterDiscard(Candidate candidate, Candidate incumbent, Mode mode) {
    if (mode == Mode.FOLD && candidate.danger != incumbent.danger) {
      return candidate.danger < incumbent.danger;
    }
    if (candidate.quality.shanten() != incumbent.quality.shanten()) {
      return candidate.quality.shanten() < incumbent.quality.shanten();
    }
    long candidateScore = candidate.score(mode);
    long incumbentScore = incumbent.score(mode);
    if (candidateScore != incumbentScore) {
      return candidateScore > incumbentScore;
    }
    if (candidate.quality.aka() != incumbent.quality.aka()) {
      return candidate.quality.aka() > incumbent.quality.aka();
    }
    if (candidate.danger != incumbent.danger) {
      return candidate.danger < incumbent.danger;
    }
    boolean candidateRed = candidate.action.selection().red();
    if (candidateRed != incumbent.action.selection().red()) {
      return !candidateRed;
    }
    boolean candidateTsumogiri = candidate.action.selection().tsumogiri();
    if (candidateTsumogiri != incumbent.action.selection().tsumogiri()) {
      return candidateTsumogiri;
    }
    return identity(candidate.action) < identity(incumbent.action);
  }

  /**
   * 鳴きを比べ、候補のほうが良ければ true を返す。比べる順は、向聴、攻めの点、残る赤の数、種別（チー・ポン・大明槓の順）、牌の種類の
   * 小さいほう、手の赤を使わないほう。
   */
  private static boolean betterCall(Candidate candidate, Candidate incumbent) {
    if (candidate.quality.shanten() != incumbent.quality.shanten()) {
      return candidate.quality.shanten() < incumbent.quality.shanten();
    }
    if (candidate.score != incumbent.score) {
      return candidate.score > incumbent.score;
    }
    if (candidate.quality.aka() != incumbent.quality.aka()) {
      return candidate.quality.aka() > incumbent.quality.aka();
    }
    if (candidate.action.type() != incumbent.action.type()) {
      return candidate.action.type().ordinal() < incumbent.action.type().ordinal();
    }
    if (candidate.action.tile() != incumbent.action.tile()) {
      return candidate.action.tile() < incumbent.action.tile();
    }
    return !candidate.action.selection().red() && incumbent.action.selection().red();
  }

  /** 槓の後の手を比べ、候補のほうが良ければ true を返す。比べる順は、向聴、攻めの点。 */
  private static boolean betterQuality(Quality candidate, Quality incumbent) {
    if (candidate.shanten() != incumbent.shanten()) {
      return candidate.shanten() < incumbent.shanten();
    }
    return offense(candidate, false) > offense(incumbent, false);
  }

  /** 打つ牌の並び順。種類の順で、同じ種類なら赤を後にする。 */
  private static int identity(Action discard) {
    return discard.tile() * 2 + (discard.selection().red() ? 1 : 0);
  }

  /** その種別の手のうち、最初のものが合法手の何番目かを返す。無ければ −1。 */
  private int find(Action.Type type) {
    for (int action = 0; action < o.legalActions(); action++) {
      if (o.action(action).type() == type) {
        return action;
      }
    }
    return -1;
  }

  private Quality nextQuality() {
    if (usedQualities == CAPACITY) {
      throw new IllegalStateException("A decision needs more than " + CAPACITY + " hand analyses");
    }
    return qualities[usedQualities++];
  }

  private Candidate nextCandidate() {
    if (usedCandidates == CAPACITY) {
      throw new IllegalStateException("A decision needs more than " + CAPACITY + " candidates");
    }
    return candidates[usedCandidates++];
  }

  /** 打牌か鳴きの候補。打牌の候補が持つ手は打牌、鳴きの候補が持つ手は鳴き。 */
  private static final class Candidate {

    /** 選ぶときの合法手の番号。鳴いた後の打牌の候補では、その鳴きの番号。 */
    private int index;

    private Action action;
    private Quality quality;
    private int danger;

    /** 押し引きに依らない攻めの点。 */
    private long score;

    private Candidate set(int index, Action action, Quality quality, int danger, long score) {
      this.index = index;
      this.action = action;
      this.quality = quality;
      this.danger = danger;
      this.score = score;
      return this;
    }

    /** その押し引きでの点を返す。押すときだけ危険度の分を引く。 */
    private long score(Mode mode) {
      return mode == Mode.PRESS ? score - danger * PRESS_DANGER_WEIGHT : score;
    }
  }
}
