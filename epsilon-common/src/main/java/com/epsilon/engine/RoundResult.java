package com.epsilon.engine;

import java.util.AbstractList;
import java.util.List;
import java.util.Objects;
import java.util.RandomAccess;

/** 和了または流局による1局の結果。供託を含む半荘上の精算は {@link RoundSettlement} が持つ。 */
public sealed interface RoundResult
    permits RoundResult.Winning, RoundResult.ExhaustiveDraw, RoundResult.AbortiveDraw {

  /**
   * 供託を除く局結果の点棒増減を返す。
   *
   * @return 四席分の点棒増減
   */
  PointDelta pointDelta();

  /** 供託を受け取る和了結果。 */
  sealed interface Winning extends RoundResult permits TsumoAgari, RonAgari {

    /**
     * 局内で成立した和了情報を優先順位順に返す。
     *
     * @return 1件のツモ、または1件以上のロン和了情報
     */
    List<? extends WinClaim> claims();

    /**
     * 供託を受け取る先頭和了者を返す。
     *
     * @return 和了者の席番号（0-3）
     */
    default int riichiStickReceiver() {
      return claims().getFirst().winner();
    }

    /**
     * 成立した和了にリーチ者が含まれるかを返す。
     *
     * @return 一人でもリーチ和了者がいれば {@code true}
     */
    default boolean hasRiichiWinner() {
      for (WinClaim claim : claims()) {
        if (claim.hasRiichi()) {
          return true;
        }
      }
      return false;
    }
  }

  /** ツモ和了。 */
  final class TsumoAgari implements Winning {

    private final WinClaim.Tsumo claim;
    private final List<WinClaim.Tsumo> claims = new TsumoClaimList();

    TsumoAgari(WinClaim.Tsumo claim) {
      this.claim = Objects.requireNonNull(claim, "claim");
    }

    /**
     * 唯一のツモ和了情報を返す。
     *
     * @return ツモ和了情報
     */
    public WinClaim.Tsumo claim() {
      return claim;
    }

    /**
     * ツモ和了者を返す。
     *
     * @return 和了者の席番号（0-3）
     */
    public int winner() {
      return claim.winner();
    }

    @Override
    public List<WinClaim.Tsumo> claims() {
      return claims;
    }

    @Override
    public PointDelta pointDelta() {
      return claim.pointDelta();
    }

    private final class TsumoClaimList extends AbstractList<WinClaim.Tsumo>
        implements RandomAccess {
      @Override
      public WinClaim.Tsumo get(int index) {
        Objects.checkIndex(index, 1);
        return claim;
      }

      @Override
      public int size() {
        return 1;
      }
    }
  }

  /** 単独または複数ロン。和了者は放銃者から見たツモ順で並ぶ。 */
  final class RonAgari implements Winning {

    private static final int MAX_RON_CLAIMS = 3;
    private final WinClaim.Ron firstClaim;
    private final WinClaim.Ron secondClaim;
    private final WinClaim.Ron thirdClaim;
    private final int claimCount;
    private final List<WinClaim.Ron> claims = new RonClaimList();
    private final PointDelta pointDelta;

    RonAgari(List<WinClaim.Ron> claims) {
      if (claims.isEmpty() || claims.size() > MAX_RON_CLAIMS) {
        throw new IllegalArgumentException("RON claim count out of range: " + claims.size());
      }
      claimCount = claims.size();
      firstClaim = Objects.requireNonNull(claims.get(0), "firstClaim");
      secondClaim = claimCount >= 2 ? Objects.requireNonNull(claims.get(1), "secondClaim") : null;
      thirdClaim = claimCount >= 3 ? Objects.requireNonNull(claims.get(2), "thirdClaim") : null;
      pointDelta = sumPointDelta();
    }

    RonAgari(WinClaim.Ron firstClaim) {
      this(firstClaim, null, null, 1);
    }

    RonAgari(WinClaim.Ron firstClaim, WinClaim.Ron secondClaim) {
      this(firstClaim, secondClaim, null, 2);
    }

    RonAgari(WinClaim.Ron firstClaim, WinClaim.Ron secondClaim, WinClaim.Ron thirdClaim) {
      this(firstClaim, secondClaim, thirdClaim, 3);
    }

    private RonAgari(
        WinClaim.Ron firstClaim,
        WinClaim.Ron secondClaim,
        WinClaim.Ron thirdClaim,
        int claimCount) {
      this.firstClaim = Objects.requireNonNull(firstClaim, "firstClaim");
      this.secondClaim =
          claimCount >= 2 ? Objects.requireNonNull(secondClaim, "secondClaim") : null;
      this.thirdClaim = claimCount >= 3 ? Objects.requireNonNull(thirdClaim, "thirdClaim") : null;
      this.claimCount = claimCount;
      pointDelta = sumPointDelta();
    }

    private PointDelta sumPointDelta() {
      int player0 = 0;
      int player1 = 0;
      int player2 = 0;
      int player3 = 0;
      for (int index = 0; index < claimCount; index++) {
        WinClaim.Ron claim = claim(index);
        PointDelta delta = claim.pointDelta();
        player0 += delta.player0();
        player1 += delta.player1();
        player2 += delta.player2();
        player3 += delta.player3();
      }
      return new PointDelta(player0, player1, player2, player3);
    }

    @Override
    public List<WinClaim.Ron> claims() {
      return claims;
    }

    /**
     * 共通の放銃者を返す。
     *
     * @return 放銃者の席番号（0-3）
     */
    public int loser() {
      return firstClaim.from();
    }

    @Override
    public PointDelta pointDelta() {
      return pointDelta;
    }

    /**
     * 指定席がロンを成立させたかを返す。
     *
     * @param player 席番号（0-3）
     * @return 和了者の一覧に含まれるなら {@code true}
     */
    public boolean hasWinner(int player) {
      for (int index = 0; index < claimCount; index++) {
        if (claim(index).winner() == player) {
          return true;
        }
      }
      return false;
    }

    private WinClaim.Ron claim(int index) {
      return switch (index) {
        case 0 -> firstClaim;
        case 1 -> secondClaim;
        case 2 -> thirdClaim;
        default -> throw new IndexOutOfBoundsException(index);
      };
    }

    private final class RonClaimList extends AbstractList<WinClaim.Ron> implements RandomAccess {
      @Override
      public WinClaim.Ron get(int index) {
        Objects.checkIndex(index, claimCount);
        return claim(index);
      }

      @Override
      public int size() {
        return claimCount;
      }
    }
  }

  /**
   * 牌山を使い切った流局。
   *
   * <dl>
   *   <dt>{@code tenpaiMask}
   *   <dd>ビット {@code player} が1なら、その席が流局時聴牌している4-ビットマスク
   *   <dt>{@code nagashiWinnerMask}
   *   <dd>ビット {@code player} が1なら、その席が流し満貫を成立させた4-ビットマスク
   *   <dt>{@code pointDelta}
   *   <dd>流し満貫またはノーテン罰符による各席の点棒増減
   * </dl>
   */
  final class ExhaustiveDraw implements RoundResult {
    private final int tenpaiMask;
    private final int nagashiWinnerMask;
    private final PointDelta pointDelta;

    public ExhaustiveDraw(int tenpaiMask, int nagashiWinnerMask, PointDelta pointDelta) {
      this.tenpaiMask = tenpaiMask;
      this.nagashiWinnerMask = nagashiWinnerMask;
      this.pointDelta = Objects.requireNonNull(pointDelta, "pointDelta");
    }

    public int tenpaiMask() {
      return tenpaiMask;
    }

    public int nagashiWinnerMask() {
      return nagashiWinnerMask;
    }

    @Override
    public PointDelta pointDelta() {
      return pointDelta;
    }

    /**
     * 指定席が流局時聴牌かを返す。
     *
     * @param player 席番号（0-3）
     * @return 聴牌なら {@code true}
     */
    public boolean isTenpai(int player) {
      return (tenpaiMask & (1 << player)) != 0;
    }

    /**
     * 指定席が流し満貫を成立させたかを返す。
     *
     * @param player 席番号（0-3）
     * @return 流し満貫の成立者なら {@code true}
     */
    public boolean isNagashiWinner(int player) {
      return (nagashiWinnerMask & (1 << player)) != 0;
    }

    /**
     * この流局に流し満貫が含まれるかを返す。
     *
     * @return 一席以上が成立させていれば {@code true}
     */
    public boolean hasNagashiMangan() {
      return nagashiWinnerMask != 0;
    }
  }

  /** 途中流局を成立させたルール。 */
  enum AbortiveDrawReason {
    NINE_TERMINALS_AND_HONORS("九種九牌"),
    TRIPLE_RON("三家和了"),
    FOUR_WINDS("四風連打"),
    FOUR_RIICHI("四家立直"),
    FOUR_KANS("四槓散了");

    private final String label;

    AbortiveDrawReason(String label) {
      this.label = label;
    }

    /** 牌譜に保存する日本語の流局理由。 */
    public String label() {
      return label;
    }
  }

  /**
   * 九種九牌などの途中流局。
   *
   * <dl>
   *   <dt>{@code reason}
   *   <dd>局を中断したルール上の理由
   * </dl>
   */
  final class AbortiveDraw implements RoundResult {

    private static final PointDelta NO_POINTS = new PointDelta(0, 0, 0, 0);
    private final AbortiveDrawReason reason;

    public AbortiveDraw(AbortiveDrawReason reason) {
      this.reason = Objects.requireNonNull(reason, "reason");
    }

    public AbortiveDrawReason reason() {
      return reason;
    }

    @Override
    public PointDelta pointDelta() {
      return NO_POINTS;
    }
  }
}
