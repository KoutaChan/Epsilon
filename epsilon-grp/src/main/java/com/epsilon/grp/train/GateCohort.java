package com.epsilon.grp.train;

import java.util.List;

/** 関門のコホート（再学習の時点で確定済みのコホートのうち、最新のもの）をどう扱うか。 */
public enum GateCohort {
  /** 範囲の最新のコホートまでを全て学習し、関門はその次のコホートの全ての半荘で測る。 */
  NEXT,
  /** 関門のコホートも、範囲の最新のコホートとして学習する。ただしその関門用の半荘は学習にも早期終了にも使わず、関門で測る。 */
  GUARDED;

  /** 指定の範囲で学習するときに、関門で測るコホート。 */
  public int cohort(RecencyWindow window) {
    return this == NEXT ? window.newest() + 1 : window.newest();
  }

  /** 関門のコホートの半荘のうち、関門で測るもの（半荘の順）。 */
  public List<GameRecord> gateGames(List<GameRecord> games) {
    if (this == NEXT) {
      return List.copyOf(games);
    }
    return games.stream().filter(game -> slice(game.gameId(), true) == GrpSlice.GUARD).toList();
  }

  /**
   * 範囲のコホートの半荘を、学習・早期終了・関門のどれに使うかに分ける。
   *
   * @param gameId 半荘 ID
   * @param gateCohort その半荘が関門のコホートのものか
   */
  GrpSlice slice(long gameId, boolean gateCohort) {
    return this == GUARDED && gateCohort ? GrpSlice.guarded(gameId) : GrpSlice.of(gameId);
  }
}
