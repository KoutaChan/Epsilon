package com.epsilon.training;

import com.epsilon.calculate.scoring.HandScore;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.config.settings.TenhouLogSettings;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.Tile;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.PointDelta;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.WinClaim;
import com.epsilon.engine.WinPayment;
import com.epsilon.util.FormatUtils;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * 天鳳互換形式の JSON 対局ログを出力するレコーダー。
 *
 * <p>Equim-chan/tensoul の convert.js と互換性のある形式で出力する。
 */
public final class TenhouLogWriter implements GameRecorder {

  // セクション: 定数
  /** ツモ切りを表す天鳳形式のシンボル値。 */
  private static final int TSUMOGIRI = 60;

  private static final String LOG_VERSION = "2.3";
  private static final String RULE_DISP = "南喰赤";
  // 結果表示用ラベル
  private static final String AGARI = "和了";
  private static final String RYUUKYOKU = "流局";
  private static final String NAGASHI_MANGAN = "流し満貫";

  // 打点表示用ラベル
  private static final String MANGAN = "満貫";
  private static final String HANEMAN = "跳満";
  private static final String BAIMAN = "倍満";
  private static final String SANBAIMAN = "三倍満";
  private static final String YAKUMAN = "役満";
  private static final String KAZOE_YAKUMAN = "数え役満";
  private static final String FU = "符";
  private static final String HAN = "飜";
  private static final String POINT = "点";
  private static final String ALL = "∀";

  // セクション: フィールド
  private final String[] playerNames;
  private final String viewerUrlPrefix;
  private final String referenceId = UUID.randomUUID().toString();
  private final List<JsonArray> roundArrays = new ArrayList<>();
  private final List<int[]> roundFinalScores = new ArrayList<>();

  // 局単位の一時バッファ
  private int[] roundInfo;
  private int[] startScores;
  private int[] finalScores = {25000, 25000, 25000, 25000};
  private final List<Integer> doraIndicators = new ArrayList<>();
  private List<Integer> uraDoraIndicators = List.of();

  @SuppressWarnings("unchecked")
  private final List<Integer>[] haipai = new List[4];

  @SuppressWarnings("unchecked")
  private final List<Object>[] draws = new List[4];

  @SuppressWarnings("unchecked")
  private final List<Object>[] dahais = new List[4];

  /** 局の結果配列。和了時は [ラベル, 得点増減の配列, 和了者情報, 得点増減の配列, 和了者情報, ...] の形式。 */
  private List<Object> resultArray;

  /** この局でリーチ供託が成立したプレイヤーを追跡する。 */
  private final boolean[] riichiAcceptedThisRound = new boolean[4];

  // セクション: コンストラクタ
  /**
   * 指定した席名で空の対局ログを生成する。
   *
   * <p>席名は生成時に取り込み、以降の呼び出し元の配列変更を記録へ反映しない。
   *
   * @param playerNames 席順どおりのプレイヤー名。不足分は既定名で補う
   * @param settings 起動時に読み込んだ共通のビューア設定
   */
  public TenhouLogWriter(String[] playerNames, TenhouLogSettings settings) {
    this.playerNames = padNames(playerNames);
    this.viewerUrlPrefix = settings.viewerUrlPrefix();
  }

  // セクション: GameRecorder 実装 — 局の開始・終了
  @Override
  public void startRound(int roundIndex, int honba, int kyotakuCount, int[] startingScores) {
    roundInfo = new int[] {roundIndex, honba, kyotakuCount};
    startScores = startingScores.clone();
    doraIndicators.clear();
    uraDoraIndicators = List.of();
    resultArray = null;
    Arrays.fill(riichiAcceptedThisRound, false);
    for (int player = 0; player < 4; player++) {
      haipai[player] = new ArrayList<>();
      draws[player] = new ArrayList<>();
      dahais[player] = new ArrayList<>();
    }
  }

  @Override
  public void endRound(GameRecorder.RoundEnd roundEnd) {
    int[] settledScores = roundEnd.finalScores();
    ArrayList<Integer> encodedUraDoraIndicators = new ArrayList<>();
    for (int physicalTileId : roundEnd.uraDoraPhysicalTileIds()) {
      encodedUraDoraIndicators.add(physicalTileIdToTenhou(physicalTileId));
    }
    uraDoraIndicators = encodedUraDoraIndicators;

    replaceResultScoreDelta(settledScores);
    // RoundEndから受け取った独立配列を所有し、以降は変更せず全体・局別の確定点で共有する。
    finalScores = settledScores;
    roundArrays.add(buildRoundArray());
    roundFinalScores.add(finalScores);
  }

  // セクション: GameRecorder 実装 — 配牌・ドラ
  @Override
  public void recordHaipai(int player, int[] concealedTileCounts, int concealedAkaMask) {
    ArrayList<Integer> encodedHand = new ArrayList<>();
    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      for (int copyIndex = 0; copyIndex < concealedTileCounts[tileType]; copyIndex++) {
        boolean isAkaTile = copyIndex == 0 && AkaTileMask.containsTile(concealedAkaMask, tileType);
        encodedHand.add(toTenhouId(tileType, isAkaTile));
      }
    }
    Collections.sort(encodedHand);
    haipai[player] = encodedHand;
  }

  @Override
  public void recordDoraIndicators(int[] physicalTileIds) {
    doraIndicators.clear();
    for (int physicalTileId : physicalTileIds) {
      doraIndicators.add(physicalTileIdToTenhou(physicalTileId));
    }
  }

  @Override
  public void recordNewDora(int physicalTileId) {
    doraIndicators.add(physicalTileIdToTenhou(physicalTileId));
  }

  // セクション: GameRecorder 実装 — ツモ・打牌
  @Override
  public void recordDraw(int player, int tileType, boolean isAkaTile) {
    draws[player].add(toTenhouId(tileType, isAkaTile));
  }

  @Override
  public void recordDahai(
      int player, int tileType, boolean isAkaTile, boolean tsumogiri, boolean riichi) {
    Object symbol = tsumogiri ? TSUMOGIRI : toTenhouId(tileType, isAkaTile);
    if (riichi) {
      dahais[player].add("r" + symbol);
    } else {
      dahais[player].add(symbol);
    }
  }

  @Override
  public void recordRiichiAccepted(int player) {
    riichiAcceptedThisRound[player] = true;
  }

  // セクション: GameRecorder 実装 — 鳴き
  /** チーを記録する。天鳳形式: {@code "c<called><hand1><hand2>"}。 */
  @Override
  public void recordChi(
      int player,
      int[] chiTileTypes,
      int calledTileType,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka) {
    int[] encodedTileIds =
        encodeChiTileIds(chiTileTypes, calledTileType, calledTileIsAka, consumedHandTileIsAka);
    draws[player].add("c" + encodedTileIds[0] + encodedTileIds[1] + encodedTileIds[2]);
  }

  /** ポンを記録する。天鳳形式: {@code "<tile>p<tile><tile>"}。 */
  @Override
  public void recordPon(
      int player,
      int tileType,
      int sourcePlayerOffset,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka) {
    int calledTileIndex = sourcePlayerOffsetToCalledTileIndex(sourcePlayerOffset);
    int[] encodedTileIds =
        encodePonTileIds(tileType, calledTileIndex, calledTileIsAka, consumedHandTileIsAka);
    StringBuilder encodedMeld = new StringBuilder();
    for (int tileIndex = 0; tileIndex < encodedTileIds.length; tileIndex++) {
      if (tileIndex == calledTileIndex) {
        encodedMeld.append('p');
      }
      encodedMeld.append(encodedTileIds[tileIndex]);
    }
    draws[player].add(encodedMeld.toString());
  }

  /** 大明槓を記録する。天鳳形式: {@code "<tile>m<tile><tile><tile>"}。 */
  @Override
  public void recordDaiminkan(
      int player,
      int tileType,
      int sourcePlayerOffset,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka) {
    int calledTileIndex = sourcePlayerOffsetToCalledTileIndex(sourcePlayerOffset);
    int calledTileMarkerIndex = calledTileIndex == 2 ? 3 : calledTileIndex;
    int[] encodedTileIds =
        encodeDaiminkanTileIds(
            tileType, calledTileMarkerIndex, calledTileIsAka, consumedHandTileIsAka);
    StringBuilder encodedMeld = new StringBuilder();
    for (int tileIndex = 0; tileIndex < encodedTileIds.length; tileIndex++) {
      if (tileIndex == calledTileMarkerIndex) {
        encodedMeld.append('m');
      }
      encodedMeld.append(encodedTileIds[tileIndex]);
    }
    draws[player].add(encodedMeld.toString());
    dahais[player].add(0);
  }

  /** 暗槓を記録する。天鳳形式: {@code "<t><t><t>a<t>"}。 */
  @Override
  public void recordAnkan(int player, int tileType, boolean containsAkaTile) {
    int[] encodedTileIds = encodeAnkanTileIds(tileType, containsAkaTile);
    dahais[player].add(
        "" + encodedTileIds[0] + encodedTileIds[1] + encodedTileIds[2] + "a" + encodedTileIds[3]);
  }

  /** 加槓を記録する。既存のポンエンコードを探し、{@code p→k} に置換して dahais に追加する。 */
  @Override
  public void recordKakan(int player, int tileType, boolean addedTileIsAka) {
    int encodedAddedTileId = toTenhouId(tileType, addedTileIsAka);
    int encodedNonAkaTileId = toTenhouId(tileType, false);
    int encodedAkaTileId = toTenhouId(tileType, true);

    for (int drawIndex = draws[player].size() - 1; drawIndex >= 0; drawIndex--) {
      Object recordedDraw = draws[player].get(drawIndex);
      if (!(recordedDraw instanceof String encodedPon)) {
        continue;
      }
      if (encodedPon.contains("p" + encodedNonAkaTileId)
          || encodedPon.contains("p" + encodedAkaTileId)) {
        dahais[player].add(encodedPon.replaceFirst("p", "k" + encodedAddedTileId));
        return;
      }
    }
    dahais[player].add("k" + encodedAddedTileId);
  }

  // セクション: GameRecorder 実装 — 局結果
  @Override
  public void recordRoundResult(RoundResult result) {
    resultArray = new ArrayList<>();
    switch (result) {
      case RoundResult.Winning winning -> {
        resultArray.add(AGARI);
        for (WinClaim claim : winning.claims()) {
          resultArray.add(toIntList(claim.pointDelta()));
          resultArray.add(buildWinnerInfo(claim));
        }
      }
      case RoundResult.ExhaustiveDraw draw -> {
        resultArray.add(draw.hasNagashiMangan() ? NAGASHI_MANGAN : RYUUKYOKU);
        resultArray.add(toIntList(draw.pointDelta()));
      }
      case RoundResult.AbortiveDraw draw -> resultArray.add(draw.reason().label());
    }
  }

  // セクション: JSON 出力
  /**
   * 全局のログを天鳳互換 JSON 文字列として出力する。
   *
   * @return 記録済み全局を含む JSON 文字列
   */
  public String toJson() {
    return buildRoot(referenceId, roundArrays, finalScores).toString();
  }

  /**
   * 指定した1局だけを天鳳互換 JSON 文字列として出力する。
   *
   * @param roundIndex 記録済み局の0始まりインデックス
   * @return 指定局だけを含む JSON 文字列
   * @throws IndexOutOfBoundsException 指定インデックスの局が記録されていない場合
   */
  public String toRoundJson(int roundIndex) {
    if (roundIndex < 0 || roundIndex >= roundArrays.size()) {
      throw new IndexOutOfBoundsException(
          "roundIndex=" + roundIndex + ", roundCount=" + roundArrays.size());
    }
    String roundRef = referenceId + "-round-" + FormatUtils.zeroPad(roundIndex + 1, 2);
    return buildRoot(
            roundRef, List.of(roundArrays.get(roundIndex)), roundFinalScores.get(roundIndex))
        .toString();
  }

  /**
   * 記録済みの局数を返す。
   *
   * @return JSON へ出力できる完了局数
   */
  public int roundCount() {
    return roundArrays.size();
  }

  /**
   * 全局ログを指定されたJSONビューアのURLとして出力する。
   *
   * @return URL 符号化済みのビューア URL
   */
  public String toViewerUrl() {
    return toViewerUrl(toJson(), viewerUrlPrefix);
  }

  /**
   * 指定した1局だけを指定されたJSONビューアのURLとして出力する。
   *
   * @param roundIndex 記録済み局の0始まりインデックス
   * @return 指定局だけを表示するビューア URL
   */
  public String toRoundViewerUrl(int roundIndex) {
    return toViewerUrl(toRoundJson(roundIndex), viewerUrlPrefix);
  }

  /**
   * 全体・局別のビューア URL を上段に、対応する JSON 本文を空行を挟んだ下段に出力する。
   *
   * @return URL 一覧と JSON 一覧を、それぞれ全体・各局の順に一行ずつ並べた文字列
   */
  public String toViewerUrlAndJsonLines() {
    String json = toJson();
    StringBuilder urlLines = new StringBuilder(toViewerUrl(json, viewerUrlPrefix));
    StringBuilder jsonLines = new StringBuilder(json);
    for (int roundIndex = 0; roundIndex < roundCount(); roundIndex++) {
      String roundJson = toRoundJson(roundIndex);
      urlLines.append(System.lineSeparator()).append(toViewerUrl(roundJson, viewerUrlPrefix));
      jsonLines.append(System.lineSeparator()).append(roundJson);
    }
    return urlLines
        .append(System.lineSeparator())
        .append(System.lineSeparator())
        .append(jsonLines)
        .toString();
  }

  /**
   * 天鳳 JSON 文字列をビューア URL に変換する。
   *
   * @param json 天鳳互換の対局ログ JSON
   * @param viewerUrlPrefix JSONを連結するビューアのURL接頭辞
   * @return 指定したビューアURLの接頭辞に、URL符号化済みのJSONを連結したURL
   */
  public static String toViewerUrl(String json, String viewerUrlPrefix) {
    String encodedJson = URLEncoder.encode(json, StandardCharsets.UTF_8).replace("+", "%20");
    return viewerUrlPrefix + encodedJson;
  }

  private JsonObject buildRoot(String referenceId, List<JsonArray> roundArrays, int[] finalScores) {
    JsonObject root = new JsonObject();
    root.addProperty("ver", LOG_VERSION);
    root.addProperty("ref", referenceId);

    JsonArray encodedRounds = new JsonArray();
    for (JsonArray roundArray : roundArrays) {
      encodedRounds.add(roundArray);
    }
    root.add("log", encodedRounds);

    root.addProperty("ratingc", "PF4");
    root.addProperty("lobby", 0);

    root.add("rule", buildRuleObject());
    addPlayerFields(root, finalScores);
    addTitleField(root);

    return root;
  }

  // セクション: 牌 ID 変換
  /** epsilon 牌種 ID を天鳳牌 ID に変換する。 */
  static int toTenhouId(int tileType, boolean isAkaTile) {
    if (isAkaTile) {
      if (tileType == Tile.M5) {
        return 51;
      }
      if (tileType == Tile.P5) {
        return 52;
      }
      if (tileType == Tile.S5) {
        return 53;
      }
    }
    if (tileType < 27) {
      int suitIndex = tileType / 9;
      int tileNumber = tileType % 9 + 1;
      return (suitIndex + 1) * 10 + tileNumber;
    }
    return 41 + (tileType - 27);
  }

  /** 実牌 ID を天鳳牌 ID に変換する。 */
  static int physicalTileIdToTenhou(int physicalTileId) {
    return toTenhouId(Tile.typeOf(physicalTileId), Tile.isAka(physicalTileId));
  }

  // セクション: 内部: 局配列の構築
  /** 天鳳形式の局配列を構築する。 */
  private JsonArray buildRoundArray() {
    JsonArray roundArray = new JsonArray();
    roundArray.add(toJsonArray(roundInfo));
    roundArray.add(toJsonArray(startScores));
    roundArray.add(toJsonArray(doraIndicators));
    roundArray.add(toJsonArray(uraDoraIndicators));
    for (int player = 0; player < 4; player++) {
      roundArray.add(toJsonArray(haipai[player]));
      roundArray.add(toMixedArray(draws[player]));
      roundArray.add(toMixedArray(dahais[player]));
    }
    roundArray.add(resultArray != null ? toResultArray(resultArray) : new JsonArray());
    return roundArray;
  }

  // セクション: 内部: 確定点の反映
  /** 確定した得点から各家の得点増減を逆算する。リーチ宣言時の1000点支払いは、打牌の r マーカーで表す。 */
  private void replaceResultScoreDelta(int[] settledScores) {
    // 途中流局では、天鳳形式の得点増減を出力しない。
    if (resultArray.size() < 2) {
      return;
    }

    ArrayList<Integer> scoreDeltas = new ArrayList<>(4);
    for (int player = 0; player < 4; player++) {
      int riichiPayment = riichiAcceptedThisRound[player] ? 1000 : 0;
      int scoreDelta = settledScores[player] - startScores[player] + riichiPayment;
      // 複数ロンでは、2人目以降の和了による得点増減を保ち、供託などの精算差分を最初の和了者の記録に反映する。
      for (int deltaIndex = 3; deltaIndex < resultArray.size(); deltaIndex += 2) {
        List<?> claimDelta = (List<?>) resultArray.get(deltaIndex);
        scoreDelta -= (Integer) claimDelta.get(player);
      }
      scoreDeltas.add(scoreDelta);
    }
    resultArray.set(1, scoreDeltas);
  }

  // セクション: 内部: 和了情報の構築
  /** 和了者情報を構築する。 */
  private List<Object> buildWinnerInfo(WinClaim claim) {
    ArrayList<Object> info = new ArrayList<>();
    info.add(claim.winner());
    info.add(claim.from());
    info.add(claim.winner());
    info.add(buildScoreDescription(claim.score(), basePayment(claim)));
    HandScore agari = claim.score();
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (!agari.hasYaku(yaku)) continue;
      info.add(formatYaku(yaku, agari.isMenzen(), claim.winner()));
    }
    appendBonus(info, "ドラ", agari.omoteDoraCount());
    appendBonus(info, "赤ドラ", agari.akaDoraCount());
    appendBonus(info, "裏ドラ", agari.uraDoraCount());
    return info;
  }

  private static void appendBonus(List<Object> info, String name, int count) {
    if (count > 0) {
      info.add(name + "(" + count + HAN + ")");
    }
  }

  /** 打点表示文字列を構築する。 */
  private String buildScoreDescription(HandScore agari, WinPayment payment) {
    String label = limitLabel(agari);
    return label != null
        ? formatScore(label, payment)
        : formatScore(agari.fu() + FU + agari.han() + HAN, payment);
  }

  private static String formatScore(String prefix, WinPayment payment) {
    return switch (payment) {
      case WinPayment.Ron ron -> prefix + ron.points() + POINT;
      case WinPayment.DealerTsumo tsumo -> prefix + tsumo.each() + POINT + ALL;
      case WinPayment.ChildTsumo tsumo ->
          prefix + tsumo.fromChild() + "-" + tsumo.fromDealer() + POINT;
    };
  }

  private static WinPayment basePayment(WinClaim claim) {
    int honbaBonus = claim.honba() * 100;
    return switch (claim.payment()) {
      case WinPayment.Ron ron -> new WinPayment.Ron(ron.points() - honbaBonus * 3);
      case WinPayment.DealerTsumo tsumo -> new WinPayment.DealerTsumo(tsumo.each() - honbaBonus);
      case WinPayment.ChildTsumo tsumo ->
          new WinPayment.ChildTsumo(
              tsumo.fromDealer() - honbaBonus, tsumo.fromChild() - honbaBonus);
    };
  }

  private static String limitLabel(HandScore agari) {
    if (agari.isYakuman()) return YAKUMAN;
    return switch (agari.basePoints()) {
      case 8000 -> KAZOE_YAKUMAN;
      case 6000 -> SANBAIMAN;
      case 4000 -> BAIMAN;
      case 3000 -> HANEMAN;
      case 2000 -> MANGAN;
      default -> null;
    };
  }

  /** 役名の表示形式。例: {@code "立直(1飜)"}, {@code "国士無双(役満)"}。 */
  private String formatYaku(ScoringYaku yaku, boolean menzen, int winner) {
    return tenhouYakuName(yaku, winner)
        + "("
        + (yaku.isYakuman() ? YAKUMAN : yaku.han(menzen) + HAN)
        + ")";
  }

  private String tenhouYakuName(ScoringYaku yaku, int winner) {
    return switch (yaku) {
      case DOUBLE_RIICHI -> "両立直";
      case TANYAO -> "断幺九";
      case YAKUHAI_HAKU -> "役牌 白";
      case YAKUHAI_HATSU -> "役牌 發";
      case YAKUHAI_CHUN -> "役牌 中";
      case YAKUHAI_SEAT -> "自風 " + Tile.name(jikazeOf(winner));
      case YAKUHAI_ROUND -> "場風 " + Tile.name(bakazeOfCurrentRound());
      case CHANTA -> "混全帯幺九";
      case JUNCHAN -> "純全帯幺九";
      default -> yaku.label();
    };
  }

  private int bakazeOfCurrentRound() {
    int windOffset = roundInfo == null ? 0 : Math.floorDiv(roundInfo[0], 4);
    return Tile.TON + Math.floorMod(windOffset, 4);
  }

  private int jikazeOf(int player) {
    int oya = roundInfo == null ? 0 : Math.floorMod(roundInfo[0], 4);
    int seatOffset = Math.floorMod(player - oya, 4);
    return Tile.TON + seatOffset;
  }

  // セクション: 内部: 鳴きエンコード
  /** チー牌を天鳳 ID 配列に変換する。 */
  private static int[] encodeChiTileIds(
      int[] chiTileTypes,
      int calledTileType,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka) {
    int calledTileId = toTenhouId(calledTileType, calledTileIsAka);
    ArrayList<Integer> consumedHandTileIds = new ArrayList<>(2);
    boolean calledTileSkipped = false;
    boolean akaHandTileEncoded = false;

    for (int tileType : chiTileTypes) {
      if (!calledTileSkipped && tileType == calledTileType) {
        calledTileSkipped = true;
        continue;
      }
      boolean encodeAsAka = consumedHandTileIsAka && !akaHandTileEncoded && Tile.canBeAka(tileType);
      if (encodeAsAka) {
        akaHandTileEncoded = true;
      }
      consumedHandTileIds.add(toTenhouId(tileType, encodeAsAka));
    }
    return new int[] {calledTileId, consumedHandTileIds.get(0), consumedHandTileIds.get(1)};
  }

  /** ポン牌を天鳳 ID 配列に変換する。 */
  private static int[] encodePonTileIds(
      int tileType, int calledTileIndex, boolean calledTileIsAka, boolean consumedHandTileIsAka) {
    int normalTileId = toTenhouId(tileType, false);
    int[] encodedTileIds = new int[] {normalTileId, normalTileId, normalTileId};
    encodedTileIds[calledTileIndex] = toTenhouId(tileType, calledTileIsAka);
    markConsumedHandAkaTile(encodedTileIds, calledTileIndex, tileType, consumedHandTileIsAka);
    return encodedTileIds;
  }

  /** 大明槓牌を天鳳 ID 配列に変換する。 */
  private static int[] encodeDaiminkanTileIds(
      int tileType,
      int calledTileMarkerIndex,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka) {
    int normalTileId = toTenhouId(tileType, false);
    int[] encodedTileIds = new int[] {normalTileId, normalTileId, normalTileId, normalTileId};
    encodedTileIds[calledTileMarkerIndex] = toTenhouId(tileType, calledTileIsAka);
    markConsumedHandAkaTile(encodedTileIds, calledTileMarkerIndex, tileType, consumedHandTileIsAka);
    return encodedTileIds;
  }

  /** 暗槓牌を天鳳 ID 配列に変換する。 */
  private static int[] encodeAnkanTileIds(int tileType, boolean containsAkaTile) {
    int normalTileId = toTenhouId(tileType, false);
    int akaTileId = toTenhouId(tileType, true);
    int[] encodedTileIds = new int[4];
    for (int tileIndex = 0; tileIndex < encodedTileIds.length; tileIndex++) {
      encodedTileIds[tileIndex] =
          tileIndex == 0 && containsAkaTile && Tile.canBeAka(tileType) ? akaTileId : normalTileId;
    }
    return encodedTileIds;
  }

  private static void markConsumedHandAkaTile(
      int[] encodedTileIds, int calledTileIndex, int tileType, boolean consumedHandTileIsAka) {
    if (!consumedHandTileIsAka || !Tile.canBeAka(tileType)) {
      return;
    }
    int akaTileId = toTenhouId(tileType, true);
    for (int tileIndex = 0; tileIndex < encodedTileIds.length; tileIndex++) {
      if (tileIndex != calledTileIndex) {
        encodedTileIds[tileIndex] = akaTileId;
        return;
      }
    }
  }

  /** 相対位置を天鳳の鳴きマーカー位置に変換する。 */
  private static int sourcePlayerOffsetToCalledTileIndex(int sourcePlayerOffset) {
    return switch (sourcePlayerOffset) {
      case 3 -> 0;
      case 2 -> 1;
      case 1 -> 2;
      default ->
          throw new IllegalArgumentException(
              "sourcePlayerOffset out of range: " + sourcePlayerOffset);
    };
  }

  // セクション: 内部: JSON ヘルパー
  private static JsonObject buildRuleObject() {
    JsonObject rule = new JsonObject();
    rule.addProperty("disp", RULE_DISP);
    rule.addProperty("aka53", 1);
    rule.addProperty("aka52", 1);
    rule.addProperty("aka51", 1);
    return rule;
  }

  /** プレイヤー情報フィールドをルートに追加する。 */
  private void addPlayerFields(JsonObject root, int[] scores) {
    JsonArray encodedNames = new JsonArray();
    JsonArray encodedRanks = new JsonArray();
    JsonArray encodedRatings = new JsonArray();
    JsonArray encodedSexes = new JsonArray();
    JsonArray encodedScores = new JsonArray();
    for (int player = 0; player < 4; player++) {
      encodedNames.add(playerNames[player]);
      encodedRanks.add("");
      encodedRatings.add(0);
      encodedSexes.add("C");
      encodedScores.add(scores[player]);
      encodedScores.add(0);
    }
    root.add("name", encodedNames);
    root.add("dan", encodedRanks);
    root.add("rate", encodedRatings);
    root.add("sx", encodedSexes);
    root.add("sc", encodedScores);
  }

  private void addTitleField(JsonObject root) {
    JsonArray title = new JsonArray();
    title.add(RULE_DISP);
    title.add(LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy/MM/dd")));
    root.add("title", title);
  }

  // セクション: 内部: ユーティリティ
  /** プレイヤー名の不足分を既定名で補い、4人分にそろえる。 */
  private static String[] padNames(String[] inputNames) {
    String[] paddedNames = {"AI", "AI", "AI", "AI"};
    for (int player = 0; player < Math.min(4, inputNames.length); player++) {
      paddedNames[player] = inputNames[player] == null ? "None" : inputNames[player];
    }
    return paddedNames;
  }

  private static JsonArray toJsonArray(int[] values) {
    JsonArray jsonArray = new JsonArray();
    for (int value : values) {
      jsonArray.add(value);
    }
    return jsonArray;
  }

  private static JsonArray toJsonArray(List<Integer> values) {
    JsonArray jsonArray = new JsonArray();
    for (int value : values) {
      jsonArray.add(value);
    }
    return jsonArray;
  }

  private static List<Integer> toIntList(PointDelta pointDelta) {
    ArrayList<Integer> values = new ArrayList<>(4);
    for (int player = 0; player < 4; player++) {
      values.add(pointDelta.get(player));
    }
    return values;
  }

  /** 整数・文字列の混合リストを JsonArray に変換する。 */
  private static JsonArray toMixedArray(List<Object> values) {
    JsonArray jsonArray = new JsonArray();
    for (Object item : values) {
      if (item instanceof Number number) {
        jsonArray.add(number);
      } else if (item instanceof String string) {
        jsonArray.add(string);
      } else {
        jsonArray.add(String.valueOf(item));
      }
    }
    return jsonArray;
  }

  /** 結果配列を JsonArray に変換する。 */
  @SuppressWarnings("unchecked")
  private static JsonArray toResultArray(List<Object> resultItems) {
    JsonArray jsonArray = new JsonArray();
    for (Object item : resultItems) {
      switch (item) {
        case String string -> jsonArray.add(string);
        case Integer number -> jsonArray.add(number);
        case List<?> subList -> jsonArray.add(toMixedArray((List<Object>) subList));
        case null, default -> jsonArray.add(String.valueOf(item));
      }
    }
    return jsonArray;
  }
}
