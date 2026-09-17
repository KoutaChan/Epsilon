package com.epsilon.major.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.WinSettlementProjection;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;

/**
 * CPU と GPU が共有する Decision 入力の特徴量、カテゴリ範囲、保存形式を定義する。
 *
 * <p>状態、合法行動、行動適用後に選べる遷移を固定ストライドのカテゴリ値フィールドと数値フィールド
 * に分離する。各カテゴリ値フィールドは自身の正確なカテゴリ数だけを持ち、その累積和を名前空間オフセットとして単一埋め込み
 * テーブルへ渡す。値0はbooleanのfalseや数値カテゴリのゼロにも使い、可変長構成要素の未使用格納位置かどうかはIDまたはPRESENT
 * フィールドの0で判定する。合法候補数と一候補あたりの遷移数だけは宣言済み容量区分に丸め、未使用格納位置をゼロ初期値のまま保持する。
 *
 * <p>{@link #fingerprint()} はフィールド順、容量区分、カテゴリ幅に加え、通し番号を格納する外部 enum の名前と順序も含む。したがって
 * フィールドの追加・並べ替えだけでなく、{@link Action.Type} などの並べ替えも変換済みデータセットとチェックポイント
 * の互換性を明示的に失わせ、意味の異なる入力を黙って読み込むことを防ぐ。
 */
public final class DecisionInputSchema {

  /** 変換済みデータセットとチェックポイントの互換性判定に使うスキーマ世代。 */
  public static final int VERSION = 21;

  /** 可変長格納位置の未使用値。カテゴリ埋め込み上の実値とはフィールドごとの符号化・復号処理で分離する。 */
  public static final int PAD_ID = 0;

  /** エンジンが一判断で生成し得る合法行動候補の最大数。 */
  public static final int MAX_LEGAL_ACTIONS = 40;

  /** 一人が保持できる副露の最大数。 */
  public static final int MAX_MELDS_PER_PLAYER = 4;

  /**
   * 4人打ちの一人あたり河上限。
   *
   * <p>通常の最大18打牌に、最大4面子ぶんの鳴き/加槓後追加打牌を足した22以下。境界検査用に24を確保し、超過は切り捨てず直ちに例外を送出する。
   */
  public static final int MAX_RIVER_EVENTS_PER_PLAYER = 24;

  /** 合法手軸で許可する密な容量区分。値は昇順でなければならない。 */
  public static final int[] LEGAL_ACTION_BUCKETS = {1, 2, 4, 8, 10, 12, 14, 16, 32, 40};

  /** 行動-遷移軸で許可する密な容量区分。値は昇順でなければならない。 */
  public static final int[] ACTION_TRANSITION_BUCKETS = {1, 4, 8, 12, 16};

  /** 行動候補適用後の、具体的かつ自己完結した遷移の種類。 */
  public enum ActionTransitionKind {
    /** PASSなど、手牌を変えずに行動候補だけで完結する遷移。 */
    IDENTITY,
    /** 通常打牌、または鳴き後の具体的な一打を適用した遷移。 */
    DISCARD,
    /** 槓を適用し、嶺上牌が未確定のまま次の抽選を待つ遷移。 */
    RINSHAN_PENDING,
    /** 和了・九種九牌など、その局を終了する遷移。 */
    TERMINAL
  }

  /** 共有の打牌採点器へ、同じ物理打牌がどの判断から生じたかを伝える文脈。 */
  public enum DiscardContext {
    /** 打牌を伴わない遷移。 */
    NONE,
    /** 通常手番からの打牌。 */
    TURN,
    /** CHI成立直後に選ぶ打牌。 */
    AFTER_CHI,
    /** PON成立直後に選ぶ打牌。 */
    AFTER_PON
  }

  /** 行動適用後に残る、RON拒否由来のフリテン。 */
  public enum RonFuritenKind {
    /** RON拒否に起因するフリテンなし。 */
    NONE,
    /** 同巡内だけ継続する一時フリテン。 */
    TEMPORARY,
    /** リーチ後の見逃しにより局終了まで継続するフリテン。 */
    RIICHI_PERSISTENT
  }

  /** 待ち別の役・公開得点を分離する和了方法。 */
  public enum WaitWinType {
    /** 他家の打牌で和了する経路。 */
    RON,
    /** 自摸牌で和了する経路。 */
    TSUMO
  }

  /** 判断境界を生んだ直近イベントの種別。{@link TurnEvent.KanKind} は別フィールドへ保持する。 */
  public enum EventType {
    /** 対応する直近イベントがない。 */
    NONE,
    /** 通常山または嶺上から牌を引いたイベント。 */
    DRAW,
    /** 牌を河へ捨てたイベント。 */
    DISCARD,
    /** 暗槓・加槓の成立前応答境界。 */
    KAN_ATTEMPT
  }

  /** 和了行動に対し、場況役を決める和了元を公開観測だけから分類した値。 */
  public enum WinConditions {
    /** 和了行動ではない。 */
    NONE,
    /** 親の配牌時和了。 */
    TENHOU,
    /** 子の第一自摸時和了。 */
    CHIIHOU,
    /** 通常の門前または副露後ツモ和了。 */
    TSUMO,
    /** 嶺上牌によるツモ和了。 */
    RINSHAN,
    /** 海底牌によるツモ和了。 */
    HAITEI,
    /** 通常のロン和了。 */
    RON,
    /** 加槓または国士の暗槓を奪うロン和了。 */
    CHANKAN,
    /** 河底牌に対するロン和了。 */
    HOUTEI
  }

  /** 一判断行に一件だけ存在する局・直近イベントのカテゴリ値フィールド。 */
  public enum RoundInt {
    /** 判断主体の絶対席。 */
    PLAYER_SEAT,
    /** 現在手番プレイヤーの自家基準相対席。 */
    CURRENT_PLAYER_RELATIVE_SEAT,
    /** 応答対象イベントを発生させたプレイヤーの相対席。 */
    SOURCE_PLAYER_RELATIVE_SEAT,
    /** 東1局を0とする局インデックス。 */
    KYOKU_INDEX,
    /** 親の自家基準相対席。 */
    DEALER_RELATIVE_SEAT,
    /** 場風の牌種。 */
    BAKAZE,
    /** 自風の牌種。 */
    JIKAZE,
    /** 本場数。 */
    HONBA,
    /** 場に供託されたリーチ棒数。 */
    KYOTAKU,
    /** 王牌を除く生牌山の残り枚数。 */
    WALL_REMAINING,
    /** 局内の手番番号。 */
    TURN_NUMBER,
    /** 全家を合計した成立済み槓数。 */
    TOTAL_KAN,
    /** 公開済みドラ表示牌数。 */
    DORA_INDICATOR_COUNT,
    /** オーラス相当の局なら1。 */
    ALL_LAST,
    /** 全員が第一打を終える前なら1。 */
    INITIAL_DISCARD_CYCLE,
    /** 自家が第一打を行う前なら1。 */
    SELF_BEFORE_FIRST_DISCARD,
    /** 鳴きで中断されていない自家第一自摸なら1。 */
    UNINTERRUPTED_FIRST_DRAW,
    /** 局中に一度でも鳴きが成立済みなら1。 */
    FIRST_TURN_CALL_OCCURRED,
    /** 現在の自摸または打牌が海底・河底候補なら1。 */
    LAST_LIVE_TILE,
    /** 直近判断境界を作ったイベント種別。 */
    EVENT_TYPE,
    /** 槓試行イベントの場合の槓種別。 */
    EVENT_KAN_KIND,
    /** イベント主体の自家基準相対席。 */
    EVENT_PLAYER_RELATIVE_SEAT,
    /** イベント対象牌の牌種。 */
    EVENT_TILE,
    /** イベント対象の物理牌が赤牌なら1。 */
    EVENT_IS_AKA,
    /** draw イベントの通常山・嶺上などの取得元。 */
    EVENT_DRAW_SOURCE,
    /** draw イベントが嶺上牌なら1。 */
    EVENT_RINSHAN,
    /** 槓ドラ表示が後続処理へ延期されているなら1。 */
    EVENT_DEFERRED_DORA
  }

  /** 一判断行に一件だけ存在する局・順位差の数値フィールド。 */
  public enum RoundFloat {
    /** 正規化した本場数。 */
    HONBA,
    /** 正規化した供託リーチ棒数。 */
    KYOTAKU,
    /** 正規化した生牌山残り枚数。 */
    WALL_REMAINING,
    /** 正規化した局内進行度。 */
    TURN,
    /** 判断主体の正規化得点。 */
    SELF_SCORE,
    /** 現在1位なら2位との差、そうでなければ1位までの負の得点差。 */
    SCORE_LEAD,
    /** 判断主体から現在1位までの正規化得点差。 */
    SCORE_TO_FIRST,
    /** 判断主体から現在4位までの正規化得点差。 */
    SCORE_TO_FOURTH
  }

  /** 自家基準の固定4席トークンに付与するカテゴリ値フィールド。 */
  public enum PlayerInt {
    /** 自家を0とする相対席。 */
    RELATIVE_SEAT,
    /** 起家基準の絶対席。 */
    ABSOLUTE_SEAT,
    /** この席の自風牌種。 */
    JIKAZE,
    /** 未宣言・宣言中・成立済みなどのリーチ状態。 */
    RIICHI_STATUS,
    /** 一発権が継続中なら1。 */
    IPPATSU,
    /** この席が第一打を行う前なら1。 */
    BEFORE_FIRST_DISCARD,
    /** 自家トークンで一時フリテン中なら1。他家では0。 */
    SELF_TEMPORARY_FURITEN,
    /** 自家トークンで捨て牌フリテン中なら1。他家では0。 */
    SELF_PERMANENT_FURITEN,
    /** 門前状態なら1。 */
    MENZEN,
    /** 現在得点から決まる順位。 */
    RANK,
    /** 暗槓を含む成立済み面子数。 */
    MELD_COUNT,
    /** 鳴いて公開された面子数。 */
    OPEN_MELD_COUNT,
    /** この席の成立済み槓数。 */
    KAN_COUNT,
    /** この席の河イベント数。 */
    RIVER_COUNT,
    /** リーチ宣言牌の河インデックス。未宣言ならパディング値。 */
    RIICHI_DECLARATION_INDEX,
    /** リーチ宣言後の打牌数。 */
    DISCARDS_AFTER_RIICHI,
    /** リーチ宣言後に行ったツモ切り数。 */
    POST_RIICHI_TSUMOGIRI_COUNT
  }

  /** 自家基準の固定4席トークンに付与する数値フィールド。 */
  public enum PlayerFloat {
    /** この席の正規化得点。 */
    SCORE,
    /** 判断主体との正規化得点差。 */
    SCORE_FROM_SELF,
    /** 河の長さから求めた局内進行度。 */
    RIVER_PROGRESS,
    /** 最大4面子に対する公開面子の割合。 */
    OPEN_MELD_FRACTION,
    /** リーチ宣言位置の局内進行度。 */
    RIICHI_DECLARATION_PROGRESS,
    /** この席が親なら1。 */
    DEALER
  }

  /** 固定34牌種トークンに付与するカテゴリ値フィールド。 */
  public enum TileInt {
    /** このトークン自身の34牌種インデックス。 */
    TILE_TYPE,
    /** 自家の手牌にある同牌種の枚数。 */
    SELF_HAND_COUNT,
    /** 自家が同牌種の赤牌を持つなら1。 */
    SELF_HAS_AKA,
    /** 手牌・河・副露・表示牌から観測できる同牌種の総枚数。 */
    VISIBLE_COUNT,
    /** 同牌種がドラ表示牌として見えている枚数。 */
    DORA_INDICATOR_MULTIPLICITY,
    /** 現在の表示牌から決まる同牌種のドラ倍率。 */
    DORA_MULTIPLICITY,
    /** 自家の河にある同牌種の枚数。 */
    SELF_RIVER_COUNT,
    /** 下家の河にある同牌種の枚数。 */
    SHIMOCHA_RIVER_COUNT,
    /** 対面の河にある同牌種の枚数。 */
    TOIMEN_RIVER_COUNT,
    /** 上家の河にある同牌種の枚数。 */
    KAMICHA_RIVER_COUNT,
    /** 同牌種が現在手牌の受入れなら1。 */
    SELF_UKEIRE,
    /** 同牌種が現在手牌の役を考慮しない形の上での待ちなら1。 */
    SELF_WAIT,
    /** 自家河に同牌種の待ちがあり捨て牌フリテン要因なら1。 */
    SELF_DISCARDED_WAIT,
    /** 同牌種が下家リーチに対する現物なら1。 */
    SHIMOCHA_RIICHI_GENBUTSU,
    /** 同牌種が対面リーチに対する現物なら1。 */
    TOIMEN_RIICHI_GENBUTSU,
    /** 同牌種が上家リーチに対する現物なら1。 */
    KAMICHA_RIICHI_GENBUTSU
  }

  /** 固定34牌種トークンに付与する数値フィールド。 */
  public enum TileFloat {
    /** 4枚に対する自家所持枚数の割合。 */
    SELF_HAND_FRACTION,
    /** 4枚に対する公開済み枚数の割合。 */
    VISIBLE_FRACTION,
    /** 4枚に対する未観測枚数の割合。 */
    UNSEEN_FRACTION,
    /** 正規化したドラ倍率。 */
    DORA_MULTIPLICITY
  }

  /** 各席24件までの時系列河イベントトークンに付与するカテゴリ値フィールド。 */
  public enum RiverInt {
    /** 捨てたプレイヤーの自家基準相対席。 */
    RELATIVE_PLAYER,
    /** その席の河内での0始まりインデックス。 */
    INDEX,
    /** 捨て牌の牌種。 */
    TILE,
    /** 捨てた物理牌が赤牌なら1。 */
    IS_AKA,
    /** ツモ切りなら1。 */
    TSUMOGIRI,
    /** リーチ宣言牌なら1。 */
    RIICHI,
    /** 他家の副露に取得された牌なら1。 */
    CALLED,
    /** 捨てられた時点の局内手番番号。 */
    TURN,
    /** 全家の河イベントを通した時系列番号。 */
    GLOBAL_SEQUENCE,
    /** パディングでない実河イベントなら1。 */
    PRESENT
  }

  /** 河イベントの席内・局内進行度を保持する数値フィールド。 */
  public enum RiverFloat {
    /** この席の河内における捨て牌位置の進行度。 */
    PLAYER_PROGRESS,
    /** 全家を通した捨て牌時系列の進行度。 */
    GLOBAL_PROGRESS
  }

  /** 各席4件までの面子トークンに付与するカテゴリ値フィールド。 */
  public enum MeldInt {
    /** 面子所有者の自家基準相対席。 */
    RELATIVE_PLAYER,
    /** その席の面子列内での0始まりインデックス。 */
    INDEX,
    /** CHI・PON・各種KANの面子種別。 */
    TYPE,
    /** 順子先頭または刻子・槓子牌種。 */
    BASE_TILE,
    /** 他家から取得した牌種。暗槓ではパディング値。 */
    CALLED_TILE,
    /** 鳴き元の相対方向。 */
    SOURCE,
    /** 赤牌が手牌・被取得牌のどちらから面子へ入ったか。 */
    AKA_SOURCE,
    /** 面子を構成する物理牌数。 */
    SIZE,
    /** 鳴き成立直前までの所有者河イベント数。 */
    CALL_AFTER_RIVER,
    /** 槓成立時点までの所有者河イベント数。 */
    KAN_AFTER_RIVER,
    /** パディングでない実面子なら1。 */
    PRESENT
  }

  /** 面子成立時点を保持する数値フィールド。 */
  public enum MeldFloat {
    /** 面子成立時点の所有者河進行度。 */
    PLAYER_PROGRESS,
    /** 槓へ変化した時点の所有者河進行度。 */
    KAN_PROGRESS
  }

  /** 行動候補だけで確定し、後続打牌によって変わらないカテゴリ値フィールド。 */
  public enum ActionInt {
    /** 372-行動空間内の安定行動 ID。 */
    ID,
    /** 打牌・CHI・RONなどの行動種別。 */
    TYPE,
    /** 方策グラフ上の大分類。 */
    GROUP,
    /** 判断の中心となる牌種。該当しない場合はパディング値。 */
    PRIMARY_TILE,
    /** 通常牌・赤牌などの物理牌選択。 */
    TILE_SELECTION,
    /** CHI順子の先頭牌種。CHI以外はパディング値。 */
    CHI_BASE,
    /** CHI順子内で被取得牌が占める位置。 */
    CHI_CALLED_POSITION,
    /** 同じ物理打牌をDAMA/RIICHI間で結ぶ識別情報 ID。 */
    DISCARD_IDENTITY,
    /** 成立面子に含まれる赤牌の由来。 */
    RESULTING_MELD_AKA_SOURCE,
    /** 行動直後に確定するリーチ状態。 */
    RESULTING_RIICHI_STATUS,
    /** 和了行動の天和・嶺上・槍槓などの文脈。 */
    WIN_CONTEXT,
    /** 裏ドラを数える資格がある和了なら1。 */
    URA_ELIGIBLE,
    /** 公開情報だけの得点射影で採用した決済仮定。 */
    SETTLEMENT_ASSUMPTION,
    /** 単独和了を仮定したときの自家予測順位。 */
    SOLE_WIN_PROJECTED_RANK,
    /** 行動により一発権が開始するなら1。 */
    STARTS_IPPATSU,
    /** 行動により既存の一発権を中断するなら1。 */
    BREAKS_IPPATSU,
    /** 行動後に打牌・嶺上抽選など何が続くか。 */
    FOLLOW_UP_KIND,
    /** 行動が局を終了するなら1。 */
    TERMINAL
  }

  /** 行動候補だけで確定する費用・和了・順位射影の数値フィールド。 */
  public enum ActionFloat {
    /** リーチ宣言で支払う正規化点数。 */
    RIICHI_DECLARATION_COST,
    /** リーチ棒支払い直後の自家正規化得点。 */
    SELF_SCORE_AFTER_DECLARATION,
    /** 宣言支払い後の1位までの正規化得点差。 */
    SCORE_TO_FIRST_AFTER_DECLARATION,
    /** 宣言支払い後の4位までの正規化得点差。 */
    SCORE_TO_FOURTH_AFTER_DECLARATION,
    /** 裏ドラを除き公開情報から確定する和了翻数。 */
    NORMALIZED_VISIBLE_HAN_WITHOUT_URA,
    /** 公開情報から確定する和了符。 */
    NORMALIZED_VISIBLE_FU,
    /** 公開情報から確定する基本点。 */
    NORMALIZED_VISIBLE_BASE_POINTS,
    /** 単独和了時に自家が受け取る最低正規化支払額。 */
    NORMALIZED_PAYMENT_FLOOR_SELF,
    /** 単独和了時の下家の最低正規化支払額。 */
    NORMALIZED_PAYMENT_FLOOR_SHIMOCHA,
    /** 単独和了時の対面の最低正規化支払額。 */
    NORMALIZED_PAYMENT_FLOOR_TOIMEN,
    /** 単独和了時の上家の最低正規化支払額。 */
    NORMALIZED_PAYMENT_FLOOR_KAMICHA,
    /** 和了時に参照し得る裏ドラ表示牌数。 */
    NORMALIZED_URA_INDICATOR_COUNT,
    /** 単独和了後の自家と1位の正規化得点差。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_SELF,
    /** 単独和了後の下家と1位の正規化得点差。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_SHIMOCHA,
    /** 単独和了後の対面と1位の正規化得点差。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_TOIMEN,
    /** 単独和了後の上家と1位の正規化得点差。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_KAMICHA
  }

  /**
   * 方策グラフ専用の行動候補の位置経路選択メタデータ。
   *
   * <p>値は埋め込みへ入れない。0は該当格納位置なし、それ以外は行動候補の位置インデックス + 1である。同じ物理打牌に属する
   * DAMA/RIICHIをネットワーク内で全候補同士比較せず、入力生成時に一度だけ解決して直接抽出する。
   */
  public enum ActionRoute {
    /** 同じ物理打牌識別情報群を代表する識別情報 ID。 */
    DISCARD_IDENTITY_REPRESENTATIVE,
    /** 同じ物理打牌識別情報群の代表行動候補の位置。 */
    DISCARD_REPRESENTATIVE_SLOT,
    /** この物理打牌に対応するDAMA 行動候補の位置。 */
    DAMA_SLOT,
    /** この物理打牌に対応するRIICHI 行動候補の位置。 */
    RIICHI_SLOT
  }

  /** 行動候補適用後の一つの具体遷移に付与するカテゴリ値フィールド。 */
  public enum ActionTransitionInt {
    /** 恒等変換・打牌・処理待ち・終端の遷移種別。 */
    KIND,
    /** 共有の打牌採点器へ渡す通常手番・CHI後・PON後の文脈。 */
    DISCARD_CONTEXT,
    /** 遷移内で実行する打牌の安定行動 ID。 */
    DISCARD_ACTION_ID,
    /** 遷移内で捨てる牌種。 */
    DISCARD_TILE,
    /** 遷移内で捨てる通常牌・赤牌の物理選択。 */
    TILE_SELECTION,
    /** 打牌後に捨て牌フリテンとなるなら1。 */
    RESULTING_DISCARD_FURITEN,
    /** RON拒否後に残る一時またはリーチ後フリテン。 */
    RESULTING_RON_FURITEN_KIND,
    /** 結果手牌で七対子・国士などの特殊手経路が利用可能なら1。 */
    SPECIAL_HANDS_AVAILABLE,
    /** パディングでない実遷移なら1。 */
    PRESENT
  }

  /** 結果手牌全体のシャンテン・受入れ・ドラ・面子状態を保持する数値フィールド。 */
  public enum ActionTransitionFloat {
    /** 通常形・七対子・国士の最小正規化シャンテン。 */
    NORMALIZED_MIN_SHANTEN,
    /** 通常4面子1雀頭形の正規化シャンテン。 */
    NORMALIZED_NORMAL_SHANTEN,
    /** 七対子形の正規化シャンテン。 */
    NORMALIZED_CHIITOI_SHANTEN,
    /** 国士無双形の正規化シャンテン。 */
    NORMALIZED_KOKUSHI_SHANTEN,
    /** 見えていない有効牌枚数の正規化合計。 */
    NORMALIZED_UKEIRE_COUNT,
    /** 受入れ牌種数の正規化値。 */
    NORMALIZED_UKEIRE_KINDS,
    /** 役を考慮しない形の上での待ち牌種数の正規化値。 */
    NORMALIZED_SHAPE_WAIT_KINDS,
    /** 現在の受入れ集合を固定し、1回以内の自摸で一枚以上引く非復元確率。 */
    CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW,
    /** 現在の受入れ集合を固定し、2回以内の自摸で一枚以上引く非復元確率。 */
    CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS,
    /** 現在の受入れ集合を固定し、3回以内の自摸で一枚以上引く非復元確率。 */
    CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS,
    /** 現在の形待ちに残る未観測物理牌枚数の正規化値。 */
    NORMALIZED_LIVE_SHAPE_WAIT_COPIES,
    /** 立直や将来イベントを仮定しない通常RONで、現在役がある待ち牌種数の正規化値。 */
    NORMALIZED_INTRINSIC_RON_YAKU_WAIT_KINDS,
    /** 立直や将来イベントを仮定しない通常RONで、現在役がある待ち残り枚数の正規化値。 */
    NORMALIZED_INTRINSIC_RON_YAKU_WAIT_COPIES,
    /** 立直や将来イベントを仮定しない通常TSUMOで、現在役がある待ち牌種数の正規化値。 */
    NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_KINDS,
    /** 立直や将来イベントを仮定しない通常TSUMOで、現在役がある待ち残り枚数の正規化値。 */
    NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_COPIES,
    /** 1シャンテンから次の自摸・打牌でフリテンでない通常RON役あり聴牌へ進める自摸牌種数。 */
    NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS,
    /** 1シャンテンから次の自摸・打牌でフリテンでない通常RON役あり聴牌へ進める残り物理牌枚数。 */
    NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_COPIES,
    /** 1シャンテンから次の自摸・打牌で、通常RON役があるがフリテンとなる聴牌にのみ進める自摸牌種数。 */
    NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_KINDS,
    /** 1シャンテンから次の自摸・打牌で、通常RON役があるがフリテンとなる聴牌にのみ進める残り物理牌枚数。 */
    NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_COPIES,
    /** 1シャンテンから次の自摸・打牌で通常TSUMO役あり聴牌へ進める自摸牌種数。 */
    NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_KINDS,
    /** 1シャンテンから次の自摸・打牌で通常TSUMO役あり聴牌へ進める残り物理牌枚数。 */
    NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_COPIES,
    /** 結果手牌・面子に含まれるドラ枚数の正規化値。 */
    NORMALIZED_DORA_COUNT,
    /** 結果手牌・面子に含まれる赤牌枚数の正規化値。 */
    NORMALIZED_AKA_TILE_COUNT,
    /** 行動候補中心牌の未観測枚数を正規化した値。 */
    NORMALIZED_PRIMARY_TILE_UNSEEN_COPIES,
    /** 結果手牌が門前なら1。 */
    MENZEN,
    /** 結果面子数の正規化値。 */
    NORMALIZED_MELD_COUNT,
    /** 結果手牌に残る副露に含まれない手牌の枚数の正規化値。 */
    NORMALIZED_CONCEALED_TILE_COUNT
  }

  /** 一牌種・一遷移に付与する、RON/TSUMO別の公開得点特徴。 */
  public enum ActionTransitionWaitFloat {
    /** RON時に公開情報から確定する正規化翻数。 */
    RON_NORMALIZED_VISIBLE_HAN,
    /** RON時に公開情報から確定する正規化符。 */
    RON_NORMALIZED_FU,
    /** RON時に公開情報から確定する正規化基本点。 */
    RON_NORMALIZED_BASE_POINTS,
    /** TSUMO時に公開情報から確定する正規化翻数。 */
    TSUMO_NORMALIZED_VISIBLE_HAN,
    /** TSUMO時に公開情報から確定する正規化符。 */
    TSUMO_NORMALIZED_FU,
    /** TSUMO時に公開情報から確定する正規化基本点。 */
    TSUMO_NORMALIZED_BASE_POINTS
  }

  /** 一行に含まれる局カテゴリ値フィールド数。 */
  public static final int ROUND_INT_COUNT = RoundInt.values().length;

  /** 一行に含まれる局数値フィールド数。 */
  public static final int ROUND_FLOAT_COUNT = RoundFloat.values().length;

  /** 一つのプレイヤートークンに含まれるカテゴリ値フィールド数。 */
  public static final int PLAYER_INT_STRIDE = PlayerInt.values().length;

  /** 一つのプレイヤートークンに含まれる数値フィールド数。 */
  public static final int PLAYER_FLOAT_STRIDE = PlayerFloat.values().length;

  /** 一つの牌トークンに含まれるカテゴリ値フィールド数。 */
  public static final int TILE_INT_STRIDE = TileInt.values().length;

  /** 一つの牌トークンに含まれる数値フィールド数。 */
  public static final int TILE_FLOAT_STRIDE = TileFloat.values().length;

  /** 一つの河トークンに含まれるカテゴリ値フィールド数。 */
  public static final int RIVER_INT_STRIDE = RiverInt.values().length;

  /** 一つの河トークンに含まれる数値フィールド数。 */
  public static final int RIVER_FLOAT_STRIDE = RiverFloat.values().length;

  /** 一つの面子トークンに含まれるカテゴリ値フィールド数。 */
  public static final int MELD_INT_STRIDE = MeldInt.values().length;

  /** 一つの面子トークンに含まれる数値フィールド数。 */
  public static final int MELD_FLOAT_STRIDE = MeldFloat.values().length;

  /** 一つの行動候補に含まれるカテゴリ値フィールド数。 */
  public static final int ACTION_INT_STRIDE = ActionInt.values().length;

  /** 一つの行動候補に含まれる数値フィールド数。 */
  public static final int ACTION_FLOAT_STRIDE = ActionFloat.values().length;

  /** 一つの行動候補に含まれる方策グラフ経路数。 */
  public static final int ACTION_ROUTE_STRIDE = ActionRoute.values().length;

  /** 一つの遷移に含まれるカテゴリ値フィールド数。 */
  public static final int ACTION_TRANSITION_INT_STRIDE = ActionTransitionInt.values().length;

  /** 一つの遷移に含まれる数値フィールド数。 */
  public static final int ACTION_TRANSITION_FLOAT_STRIDE = ActionTransitionFloat.values().length;

  /** 一つの遷移が持つ牌種別行動適用後の状態特徴量数。 */
  public static final int ACTION_TRANSITION_TILE_COUNT = Tile.NUM_TILE_TYPES;

  /** 国士無双13面待ちを含む、一つの行動適用後の状態が持ち得る最大待ち牌種数。 */
  public static final int MAX_WAIT_TILE_TYPES = 13;

  /** 遷移-牌符号化・復号処理がまとめる6個のboolean フラグ組合せ数。 */
  public static final int ACTION_TRANSITION_TILE_FLAG_COMBINATION_COUNT = 1 << 6;

  /** 手牌枚数・可視枚数・6 フラグを一つのカテゴリへ連結した辞書幅。 */
  public static final int ACTION_TRANSITION_TILE_DICTIONARY_SIZE =
      (Tile.TILES_PER_TYPE + 1)
              * (Tile.TILES_PER_TYPE + 1)
              * ACTION_TRANSITION_TILE_FLAG_COMBINATION_COUNT
          + 1;

  /** 一つのカテゴリ IDへ連結する役マスクのビット数。 */
  public static final int WAIT_YAKU_BITS_PER_CHUNK = 14;

  /** RONまたはTSUMO一経路の全役を保持するために必要なまとまり数。 */
  public static final int WAIT_YAKU_CHUNKS_PER_WIN_TYPE =
      (ScoringYaku.values().length + WAIT_YAKU_BITS_PER_CHUNK - 1) / WAIT_YAKU_BITS_PER_CHUNK;

  /** 一待ち格納位置に並べるRON・TSUMO両経路の役まとまり数。 */
  public static final int ACTION_TRANSITION_WAIT_YAKU_STRIDE = WAIT_YAKU_CHUNKS_PER_WIN_TYPE * 2;

  /** 役ビットまとまりをパディング ID込みで埋め込みする辞書幅。 */
  public static final int ACTION_TRANSITION_WAIT_YAKU_DICTIONARY_SIZE =
      (1 << WAIT_YAKU_BITS_PER_CHUNK) + 1;

  /** 一待ち格納位置に含まれるRON・TSUMO別得点フィールド数。 */
  public static final int ACTION_TRANSITION_WAIT_FLOAT_STRIDE =
      ActionTransitionWaitFloat.values().length;

  /** 埋め込み辞書を通さない公開履歴関係の開始位置。ここまでは従来のカテゴリ値フィールド。 */
  public static final int PUBLIC_HISTORY_RELATION_OFFSET =
      ROUND_INT_COUNT
          + GameState.NUM_PLAYERS * PLAYER_INT_STRIDE
          + Tile.NUM_TILE_TYPES * TILE_INT_STRIDE
          + GameState.NUM_PLAYERS * MAX_RIVER_EVENTS_PER_PLAYER * RIVER_INT_STRIDE
          + GameState.NUM_PLAYERS * MAX_MELDS_PER_PLAYER * MELD_INT_STRIDE;

  /** [query4, publicToken112, packedWord2]の未加工の short メタデータ要素数。 */
  public static final int PUBLIC_HISTORY_RELATION_COUNT =
      DecisionPublicHistory.QUERY_COUNT
          * DecisionPublicHistory.TOKEN_COUNT
          * DecisionPublicHistory.PACKED_WORDS;

  /** カテゴリ値部分と未加工の公開履歴メタデータを合わせた一行のshort 連続バッファ要素数。 */
  public static final int STATE_INT_COUNT =
      PUBLIC_HISTORY_RELATION_OFFSET + PUBLIC_HISTORY_RELATION_COUNT;

  /** 一行の状態数値連続バッファ要素数。 */
  public static final int STATE_FLOAT_COUNT =
      ROUND_FLOAT_COUNT
          + GameState.NUM_PLAYERS * PLAYER_FLOAT_STRIDE
          + Tile.NUM_TILE_TYPES * TILE_FLOAT_STRIDE
          + GameState.NUM_PLAYERS * MAX_RIVER_EVENTS_PER_PLAYER * RIVER_FLOAT_STRIDE
          + GameState.NUM_PLAYERS * MAX_MELDS_PER_PLAYER * MELD_FLOAT_STRIDE;

  /**
   * フィールド順・外部enum順・容量区分幅を含むスキーマ互換性識別子を返す。
   *
   * <p>カテゴリ 配置はこのクラスの定数を参照するため、クラス初期化中には計算しない。遅延保持クラスにより、どちらのクラスが先に参照されても完全に初期化された配置だけを使用する。
   */
  public static String fingerprint() {
    return FingerprintHolder.VALUE;
  }

  private DecisionInputSchema() {}

  /**
   * 実合法候補数を収容できる最小の宣言済み行動容量区分幅を返す。
   *
   * @param legalActionCount 一判断の実合法行動数
   * @return {@link #LEGAL_ACTION_BUCKETS}内の最小収容幅
   */
  public static int legalActionBucket(int legalActionCount) {
    if (legalActionCount < 1 || legalActionCount > MAX_LEGAL_ACTIONS) {
      throw new IllegalArgumentException("legalActionCount out of range: " + legalActionCount);
    }
    return smallestBucketAtLeast(LEGAL_ACTION_BUCKETS, legalActionCount, "legal actions");
  }

  /**
   * 一候補あたりの遷移数を収容できる最小の宣言済み容量区分幅を返す。
   *
   * @param actionTransitionCount 一行動が持つ実遷移数
   * @return {@link #ACTION_TRANSITION_BUCKETS}内の最小収容幅
   */
  public static int actionTransitionBucket(int actionTransitionCount) {
    if (actionTransitionCount < 1) {
      throw new IllegalArgumentException("actionTransitionCount must be positive");
    }
    return smallestBucketAtLeast(
        ACTION_TRANSITION_BUCKETS, actionTransitionCount, "action transitions");
  }

  private static int smallestBucketAtLeast(int[] buckets, int required, String label) {
    for (int candidate : buckets) {
      if (required <= candidate) {
        return candidate;
      }
    }
    throw new IllegalArgumentException(label + " exceed schema maximum: " + required);
  }

  private static String createFingerprint() {
    String descriptor =
        "v="
            + VERSION
            + ";roundI="
            + Arrays.toString(RoundInt.values())
            + ";roundF="
            + Arrays.toString(RoundFloat.values())
            + ";playerI="
            + Arrays.toString(PlayerInt.values())
            + ";playerF="
            + Arrays.toString(PlayerFloat.values())
            + ";tileI="
            + Arrays.toString(TileInt.values())
            + ";tileF="
            + Arrays.toString(TileFloat.values())
            + ";riverI="
            + Arrays.toString(RiverInt.values())
            + ";riverF="
            + Arrays.toString(RiverFloat.values())
            + ";meldI="
            + Arrays.toString(MeldInt.values())
            + ";meldF="
            + Arrays.toString(MeldFloat.values())
            + ";actionI="
            + Arrays.toString(ActionInt.values())
            + ";actionF="
            + Arrays.toString(ActionFloat.values())
            + ";actionTransitionI="
            + Arrays.toString(ActionTransitionInt.values())
            + ";actionTransitionF="
            + Arrays.toString(ActionTransitionFloat.values())
            + ";actionTransitionWaitF="
            + Arrays.toString(ActionTransitionWaitFloat.values())
            + ";actionTransitionKinds="
            + Arrays.toString(ActionTransitionKind.values())
            + ";discardContexts="
            + Arrays.toString(DiscardContext.values())
            + ";ronFuritenKinds="
            + Arrays.toString(RonFuritenKind.values())
            + ";waitWinTypes="
            + Arrays.toString(WaitWinType.values())
            + ";actionTypes="
            + Arrays.toString(Action.Type.values())
            + ";actionGroups="
            + Arrays.toString(Action.Group.values())
            + ";tileSelections="
            + Arrays.toString(Action.TileSelection.values())
            + ";meldTypes="
            + Arrays.toString(Meld.Type.values())
            + ";meldSources="
            + Arrays.toString(Meld.RelativeSource.values())
            + ";meldAkaSources="
            + Arrays.toString(Meld.AkaSource.values())
            + ";actionFollowUps="
            + Arrays.toString(ActionEffect.NextStep.values())
            + ";settlementAssumptions="
            + Arrays.toString(WinSettlementProjection.SettlementAssumption.values())
            + ";eventTypes="
            + Arrays.toString(EventType.values())
            + ";riichiStatuses="
            + Arrays.toString(RiichiState.values())
            + ";winContexts="
            + Arrays.toString(WinConditions.values())
            + ";turnEventDrawSources="
            + Arrays.toString(TurnEvent.DrawSource.values())
            + ";turnEventKanKinds="
            + Arrays.toString(TurnEvent.KanKind.values())
            + ";legalActionBuckets="
            + Arrays.toString(LEGAL_ACTION_BUCKETS)
            + ";actionTransitionBuckets="
            + Arrays.toString(ACTION_TRANSITION_BUCKETS)
            + ";actionTransitionTileDictionary="
            + ACTION_TRANSITION_TILE_DICTIONARY_SIZE
            + ";maxWaitTileTypes="
            + MAX_WAIT_TILE_TYPES
            + ";waitYakuMaskPerChunk="
            + WAIT_YAKU_BITS_PER_CHUNK
            + ";waitYakuChunks="
            + ACTION_TRANSITION_WAIT_YAKU_STRIDE
            + ";waitYakuDictionary="
            + ACTION_TRANSITION_WAIT_YAKU_DICTIONARY_SIZE
            + ";waitFloatStride="
            + ACTION_TRANSITION_WAIT_FLOAT_STRIDE
            + ";yakus="
            + Arrays.toString(ScoringYaku.values())
            + ";categoryLayout="
            + DecisionCategoryLayout.descriptor()
            + ";tensorLayout="
            + DecisionInputLayout.descriptor()
            + ";river="
            + MAX_RIVER_EVENTS_PER_PLAYER
            + ";meld="
            + MAX_MELDS_PER_PLAYER
            + ";publicHistory="
            + DecisionPublicHistory.descriptor();
    try {
      byte[] digest =
          MessageDigest.getInstance("SHA-256").digest(descriptor.getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(digest, 0, 8);
    } catch (NoSuchAlgorithmException impossible) {
      throw new ExceptionInInitializerError(impossible);
    }
  }

  private static final class FingerprintHolder {
    private static final String VALUE = createFingerprint();
  }
}
