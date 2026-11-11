package com.epsilon.classic.input;

import com.epsilon.mahjong.tile.Tiles;

/**
 * classic の語彙。どの列挙も、埋め込みの表の行か、コードのビットの位置を決める。そのため列挙の名前・定数・並びは変換した重みとの
 * 契約になり、テストで固定している。エンジンの値は {@link VocabularyCodec} で語彙に変換する。フィールドには ordinal ＋ 1 を書き、
 * 0 はパディングに使う。
 */
public final class ClassicVocabulary {

  /** 順子の枚数。チーの鳴いた位置の数でもある。 */
  static final int RUN = 3;

  /** 1 色の中で順子の始まりになれる数字の数（1〜7 の 7 通り）。 */
  static final int RUN_STARTS = Tiles.SUIT_SIZE - RUN + 1;

  private ClassicVocabulary() {}

  /** 行動の種類。各定数は、方策のグラフでの大分類と、行動の後に続く処理を持つ。 */
  public enum ActionType {
    /** 立直の宣言でない打牌。 */
    DAHAI(ActionGroup.DAHAI, NextStep.DECISION_COMPLETE),
    CHI(ActionGroup.MELD, NextStep.IMMEDIATE_DISCARD),
    PON(ActionGroup.MELD, NextStep.IMMEDIATE_DISCARD),
    DAIMINKAN(ActionGroup.MELD, NextStep.RINSHAN_DRAW),
    ANKAN(ActionGroup.KAN, NextStep.RINSHAN_DRAW),
    KAKAN(ActionGroup.KAN, NextStep.RINSHAN_DRAW),
    RIICHI_DAHAI(ActionGroup.RIICHI, NextStep.DECISION_COMPLETE),
    TSUMO_AGARI(ActionGroup.TSUMO, NextStep.ROUND_COMPLETE),
    RON_AGARI(ActionGroup.RON, NextStep.ROUND_COMPLETE),
    PASS(ActionGroup.PASS, NextStep.DECISION_COMPLETE),
    KYUSHU_KYUHAI(ActionGroup.KYUSHU, NextStep.ROUND_COMPLETE);

    private final ActionGroup group;
    private final NextStep next;

    ActionType(ActionGroup group, NextStep next) {
      this.group = group;
      this.next = next;
    }

    public ActionGroup group() {
      return group;
    }

    /** 行動の後に続く処理。 */
    public NextStep next() {
      return next;
    }

    /** 面子を作るか、ポンを加槓にする行動（鳴きと槓）か。どれも一発を消す。 */
    public boolean createsMeld() {
      return group == ActionGroup.MELD || group == ActionGroup.KAN;
    }
  }

  /** 方策のグラフの大分類。 */
  public enum ActionGroup {
    DAHAI(false),
    /** チー・ポン・大明槓。 */
    MELD(true),
    /** 暗槓・加槓。 */
    KAN(false),
    /** 立直の宣言の打牌。 */
    RIICHI(false),
    TSUMO(false),
    RON(true),
    PASS(true),
    KYUSHU(false);

    private final boolean response;

    ActionGroup(boolean response) {
      this.response = response;
    }

    /** 他家の打牌か加槓に応答する手か。 */
    public boolean response() {
      return response;
    }
  }

  /** どの物理牌を使うか。打牌では赤かどうかとツモ切りかどうかの組み合わせを、チー・ポンでは手の赤 5 を使うかどうかを表す。 */
  public enum TileSelection {
    NOT_APPLICABLE,
    NON_AKA,
    AKA,
    TSUMOGIRI,
    TSUMOGIRI_AKA
  }

  public enum MeldType {
    CHI,
    PON,
    DAIMINKAN,
    ANKAN,
    KAKAN
  }

  /** 鳴いた相手が、自分から見てどの席か。暗槓は {@code NONE}。 */
  public enum MeldSource {
    NONE,
    SHIMOCHA,
    TOIMEN,
    KAMICHA
  }

  /** 面子の赤 5 がどこから来たか。 */
  public enum MeldAkaSource {
    NONE,
    CALLED_TILE,
    CONSUMED_HAND_TILE,
    ADDED_KAN_TILE
  }

  public enum RiichiStatus {
    NONE,
    RIICHI,
    DOUBLE_RIICHI
  }

  /** 槍槓の応答を求めた槓の種類。 */
  public enum KanKind {
    ANKAN,
    KAKAN,
    DAIMINKAN
  }

  /** ツモの出どころ。 */
  public enum DrawSource {
    WALL,
    /** 嶺上牌。 */
    RINSHAN
  }

  /** 行動の後に続く処理。 */
  public enum NextStep {
    /** 判断が終わる。 */
    DECISION_COMPLETE,
    /** チー・ポンの後に同じ席が打牌を選ぶ。 */
    IMMEDIATE_DISCARD,
    /** 槓の後に嶺上牌を引く。 */
    RINSHAN_DRAW,
    /** 和了か途中流局で局が終わる。 */
    ROUND_COMPLETE
  }

  /** 和了の後の順位を求めるときに置く、精算の仮定。 */
  public enum SettlementAssumption {
    /** ツモ。本場と場の供託を含めて、精算が 1 通りに決まる。 */
    EXACT_TSUMO,
    /** ロンで和了するのが 1 人だけで、その人が本場と場の供託を全て受け取る。 */
    SOLE_RON
  }

  /** この判断の直前に起きたイベントの種類。 */
  public enum EventType {
    NONE,
    /** ツモ（嶺上を含む）。 */
    DRAW,
    DISCARD,
    /** 加槓の宣言。この後、他家に槍槓の応答を求める。 */
    KAN_ATTEMPT
  }

  /** 和了の状況。偶然役（天和・嶺上・海底など）が付くかを決める。和了でない行動は {@code NONE}。 */
  public enum WinConditions {
    NONE,
    /** 親の第一ツモ。 */
    TENHOU,
    /** 子の第一ツモ。 */
    CHIIHOU,
    TSUMO,
    /** 嶺上牌のツモ。 */
    RINSHAN,
    /** 山の最後の牌のツモ。 */
    HAITEI,
    RON,
    /** 加槓の牌へのロン。 */
    CHANKAN,
    /** 山が尽きた後の最後の打牌へのロン。 */
    HOUTEI
  }

  /** 役。役の集合は long で持ち、ビット k が ordinal k の役を表す。待ちの役のフィールドには、この long を 14 ビットずつに分けて書く。 */
  public enum Yaku {
    RIICHI,
    DOUBLE_RIICHI,
    IPPATSU,
    MENZEN_TSUMO,
    PINFU,
    TANYAO,
    YAKUHAI_HAKU,
    YAKUHAI_HATSU,
    YAKUHAI_CHUN,
    YAKUHAI_SEAT,
    YAKUHAI_ROUND,
    RINSHAN,
    CHANKAN,
    HAITEI,
    HOUTEI,
    IPEIKOU,
    SANSHOKU,
    SANSHOKU_DOUKOU,
    ITTSU,
    TOITOI,
    SANANKOU,
    SANKANTSU,
    SHOUSANGEN,
    HONROUTOU,
    CHANTA,
    RYANPEIKOU,
    JUNCHAN,
    HONITSU,
    CHINITSU,
    CHIITOITSU,
    KOKUSHI,
    SUUANKOU,
    CHUUREN,
    DAISANGEN,
    TSUUIISOU,
    RYUUIISOU,
    CHINROUTOU,
    DAISUUSHII,
    SHOUSUUSHII,
    SUUKANTSU,
    TENHOU,
    CHIIHOU
  }

  /** 合法手の後の 1 つの遷移の種類。 */
  public enum TransitionKind {
    /** 手を変えない（見送り）。 */
    IDENTITY,
    /** 打牌（鳴いた後の打牌を含む）。 */
    DISCARD,
    /** 槓の後で、嶺上牌を引く前。 */
    RINSHAN_PENDING,
    /** 局が終わる（和了と九種九牌）。 */
    TERMINAL
  }

  /** 遷移で打つ牌を、どの場面で選ぶか。 */
  public enum DiscardContext {
    /** 打牌の無い遷移。 */
    NONE,
    /** ツモの後の手番。 */
    TURN,
    AFTER_CHI,
    AFTER_PON
  }

  /** 遷移の後に残る、ロンの見逃しの振聴。 */
  public enum RonFuritenKind {
    NONE,
    /** 同巡の間だけ続く。 */
    TEMPORARY,
    /** 立直の後の見逃しで、局の終わりまで続く。 */
    RIICHI_PERSISTENT
  }

  /** 待ちの役と点をロンとツモに分けて持つときの、和了の仕方。 */
  public enum WaitWinType {
    RON,
    TSUMO
  }

  /**
   * 行動の番号（372 通り）を分けたブロック。各定数は幅を持ち、ブロックの先頭の番号は、幅を宣言の順に足し上げて最初に一度だけ求める。
   * ブロックの中は牌の種類の順に並べ、赤 5 は色の順（萬・筒・索）に並べる。チーは色、順子の始まり、鳴いた位置の順に並べる。
   */
  public enum ActionBlock {
    DAHAI(Tiles.KINDS),
    DAHAI_TSUMOGIRI(Tiles.KINDS),
    DAHAI_AKA(Tiles.NUMBER_SUITS),
    DAHAI_TSUMOGIRI_AKA(Tiles.NUMBER_SUITS),
    /** 1 色につき、順子の始まり 7 通り × 鳴いた位置 3 通り。 */
    CHI(Tiles.NUMBER_SUITS * RUN_STARTS * RUN),
    /**
     * 手の赤 5 を使えるチー（順子が 5 を含み、鳴いた牌が 5 でない）。1 色につき、5 を含む順子 3 通り × 5 でない鳴いた位置 2 通り。
     */
    CHI_AKA(Tiles.NUMBER_SUITS * RUN * (RUN - 1)),
    PON(Tiles.KINDS),
    PON_AKA(Tiles.NUMBER_SUITS),
    DAIMINKAN(Tiles.KINDS),
    ANKAN(Tiles.KINDS),
    KAKAN(Tiles.KINDS),
    RIICHI(Tiles.KINDS),
    RIICHI_TSUMOGIRI(Tiles.KINDS),
    RIICHI_AKA(Tiles.NUMBER_SUITS),
    RIICHI_TSUMOGIRI_AKA(Tiles.NUMBER_SUITS),
    TSUMO(1),
    RON(1),
    PASS(1),
    KYUSHU_KYUHAI(1);

    private static final int[] STARTS = starts();

    private final int width;

    ActionBlock(int width) {
      this.width = width;
    }

    public int width() {
      return width;
    }

    public int start() {
      return STARTS[ordinal()];
    }

    public int end() {
      return STARTS[ordinal() + 1];
    }

    private static int[] starts() {
      ActionBlock[] blocks = values();
      int[] starts = new int[blocks.length + 1];
      for (ActionBlock block : blocks) {
        starts[block.ordinal() + 1] = starts[block.ordinal()] + block.width;
      }
      return starts;
    }
  }

  /** 行動の番号の数（372）。 */
  public static final int ACTION_SPACE = ActionBlock.KYUSHU_KYUHAI.end();

  /** 打牌と立直の打牌が共有する、物理の打牌の番号の数（打牌の 4 つのブロックの幅の和で 74）。 */
  public static final int DISCARD_IDENTITIES = ActionBlock.DAHAI_TSUMOGIRI_AKA.end();
}
