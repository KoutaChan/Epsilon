package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.util.SeedMixer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 牌譜の半荘を、GRP の学習に使う記録（{@link GameRecord}）にする。半荘が続く局末の卓には、次の局の始まりの卓（局・本場・供託・点）
 * を使う。最終の点には、牌譜にある終局の点を使う。半荘 ID は牌譜の id のハッシュなので、学習用か検証用か（{@link GrpSlice}）は
 * 牌譜ごとに決まる。
 */
public final class LogGames {

  /** 牌譜から学ぶときのコホートの番号。学習に使う最初のコホートにする。 */
  public static final int COHORT = RecencyWindow.SKIPPED_COHORTS + 1;

  /** 牌譜のコホートに付ける Φ の項目の id。牌譜の半荘は Φ を使って集めたものではないので、この id で区別する。 */
  public static final String PHI_ID = "records";

  private static final long FNV_OFFSET = 0xcbf2_9ce4_8422_2325L;
  private static final long FNV_PRIME = 0x0000_0100_0000_01b3L;

  private LogGames() {}

  /** 牌譜の半荘 1 つを記録にする。 */
  public static GameRecord of(RecordedGame game) {
    List<RecordedRound> rounds = game.rounds();
    List<GrpState> handEnds = new ArrayList<>(rounds.size() - 1);
    for (int index = 1; index < rounds.size(); index++) {
      RoundStart next = rounds.get(index).start();
      handEnds.add(new GrpState(game.variant(), next.round(), next.honba(), next.kyotaku(), next.scores()));
    }
    return new GameRecord(game.variant(), gameId(game.meta().id()), handEnds, game.finalScores());
  }

  /**
   * 牌譜の半荘を 1 つのコホート（{@link #COHORT}）にまとめる。半荘は渡した順に並べる。半荘が無ければ IllegalArgumentException を
   * 投げる。
   */
  public static GameCohort cohort(List<RecordedGame> games) {
    if (games.isEmpty()) {
      throw new IllegalArgumentException("A cohort of records needs at least one game");
    }
    List<GameRecord> records = new ArrayList<>(games.size());
    for (RecordedGame game : games) {
      records.add(of(game));
    }
    return new GameCohort(games.getFirst().variant(), COHORT, PHI_ID, records);
  }

  /** 牌譜の id から作る 64 ビットのハッシュ。UTF-8 のバイト列の FNV-1a を、SeedMixer でさらに混ぜる。 */
  public static long gameId(String recordId) {
    long hash = FNV_OFFSET;
    for (byte value : recordId.getBytes(StandardCharsets.UTF_8)) {
      hash = (hash ^ (value & 0xff)) * FNV_PRIME;
    }
    return SeedMixer.mix64(hash);
  }
}
