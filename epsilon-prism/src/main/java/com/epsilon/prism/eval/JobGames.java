package com.epsilon.prism.eval;

import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * 評価のジョブの games.tsv のリーダー。job.json に書いた済んだ牌山の行だけを読み、牌山ごとの値（{@link WallValues}）にする。行を書いた後、
 * job.json を書く前に落ちたチャンクの行は続きが打ち直すので読まない。そのため、回っている途中のジョブも読める。
 */
public final class JobGames {

  private JobGames() {}

  /**
   * ジョブのディレクトリの job.json を読んでから、済んだ牌山の値を読む。
   *
   * @param home ジョブのディレクトリ
   * @param duel ジョブの決闘の計画
   * @return 計画の最初の牌山から、済んだ牌山までの値
   * @throws IOException job.json か games.tsv が読めないか、games.tsv の行が済んだ牌山の分より少ないとき
   * @throws IllegalArgumentException 行が計画の牌山・割り当てと合わないとき
   */
  public static WallValues read(Path home, DuelSetup duel) throws IOException {
    return read(home, RecordFile.read(JobFile.class, home.resolve(Jobs.JOB)).value(), duel);
  }

  /**
   * 済んだ牌山の値を読む。
   *
   * @param home ジョブのディレクトリ
   * @param job ジョブの記録（済んだ牌山の数）
   * @param duel ジョブの決闘の計画
   * @return 計画の最初の牌山から、済んだ牌山までの値
   * @throws IOException games.tsv が読めないか、行が済んだ牌山の分より少ないとき
   * @throws IllegalArgumentException 行が計画の牌山・割り当てと合わないとき
   */
  public static WallValues read(Path home, JobFile job, DuelSetup duel) throws IOException {
    Path file = home.resolve(JobOutputs.GAMES);
    List<String> lines = DurableFiles.lines(file);
    int perWall = duel.layout().gamesPerWall();
    long games = job.walls() * perWall;
    if (lines.size() < games) {
      throw new IOException(
          file + " holds " + lines.size() + " games, but " + job.name() + " recorded " + job.walls() + " walls of "
              + perWall + " games");
    }
    return WallValues.of(duel, lines.subList(0, (int) games));
  }
}
