package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GruShape;
import com.epsilon.grp.GruSinkhorn;
import com.epsilon.grp.GruTensor;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeTensor;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.HandResult;
import com.epsilon.grp.train.PhiStamp;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.FourPlayerFormat;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * grp のテスト用のヘルパー。ルールどおりに打った合成の半荘とその数のファイル、乱数の重みを持つ小さな GrpNetwork・GruSinkhorn・q の
 * checkpoint、コホートの確定と半荘の記録、run の設定の行と state.json、READY の候補の行を作る。
 */
final class GrpFixtures {
  static final RankUtility UTILITY = RankUtility.TENHOU;
  static final Variant VARIANT = FourPlayerFormat.VARIANT;

  /** 和了の基本点の候補（この中から一様に選ぶ）。 */
  private static final BaseClass[] BASES = {
    BaseClass.B320, BaseClass.B640, BaseClass.B1280, BaseClass.MANGAN, BaseClass.HANEMAN
  };

  private GrpFixtures() {}

  /** 半荘の番号 {@code first} から始まる、合成の半荘の記録。局の結果は荒牌・ロン・ツモを乱数で選び、Settlement と Progression のとおりに進める。 */
  static List<GameRecord> games(long first, int count, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) games.add(game(first + game, random, null));
    return games;
  }

  /**
   * 合成の半荘と、同じ局を数えた数のファイルの中身。
   *
   * @param games 合成の半荘
   * @param counted 全ての局（半荘が終わった局も）の始まりの卓と、生の結果のコード
   */
  record Played(List<GameRecord> games, CountedHands counted) {}

  /** {@link #games} と同じ半荘（同じ引数なら同じ記録）と、コホート {@code t} の数のファイルの中身を作る。 */
  static Played played(int t, long first, int count, long seed) {
    return played(t, first, count, seed, new PhiStamp("GRP", Double.NaN));
  }

  /** {@link #played(int, long, int, long)} と同じ半荘で、数のファイルのヘッダーに、収集で使った値の印 {@code stamp} を書いたもの。 */
  static Played played(int t, long first, int count, long seed, PhiStamp stamp) {
    SplittableRandom random = new SplittableRandom(seed);
    CountedHands.Builder counts = CountedHands.builder(VARIANT, t, 0, stamp);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) games.add(game(first + game, random, counts));
    return new Played(games, counts.build());
  }

  /** コホート {@code t} を確定し、項目 {@code entry} で集めた半荘の記録と、その数のファイルを書く。 */
  static void recordCounted(GrpFiles files, int t, PhiEntry entry, Played played) throws IOException {
    record(files, t, entry, played.games());
    CountFile.write(files.counts(t), played.counted());
  }

  private static GameRecord game(long gameId, SplittableRandom random, CountedHands.Builder counts) {
    TableState table = new TableState(VARIANT);
    TableState next = new TableState(VARIANT);
    RoundOutcome outcome = new RoundOutcome(VARIANT);
    int[] delta = new int[4];
    List<GrpState> handEnds = new ArrayList<>();
    while (true) {
      int kind = random.nextInt(10);
      int winner = random.nextInt(4);
      int base = BASES[random.nextInt(BASES.length)].base();
      if (kind < 2) outcome.setExhaustive(random.nextInt(16), 0, 0);
      else if (kind < 6) outcome.setRon(base, winner, (winner + 1 + random.nextInt(3)) % 4, RoundOutcome.NO_SEAT, 0);
      else outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, 0);
      if (counts != null) counts.add(gameId, table, HandResult.raw(outcome, table.dealer()));
      Settlement.apply(VARIANT, table, outcome, delta);
      Progression.next(table, outcome, delta, next);
      if (next.gameOver()) {
        int[] scores = new int[4];
        for (int seat = 0; seat < 4; seat++) scores[seat] = next.score(seat);
        return new GameRecord(VARIANT, gameId, handEnds, scores);
      }
      handEnds.add(GrpState.of(next));
      table.copyFrom(next);
    }
  }

  /** 乱数の重みを持つ GrpNetwork（学び直しと同じ形）を {@code directory} に書き、正規化した絶対パスを返す。 */
  static Path network(Path directory, long seed) throws IOException {
    return network(directory, GrpRefitter.shape(GrpFixtures.VARIANT), seed);
  }

  static Path network(Path directory, GrpShape shape, long seed) throws IOException {
    return network(directory, shape, seed, 0.1);
  }

  /**
   * 重みの標準偏差を {@code scale} にした GrpNetwork（学び直しと同じ形）を作る。{@code scale} が 1 なら局末の値が大きく外れた Φ に、0 なら
   * 全ての順列を同じ確率とする Φ（点数を動かしても値が変わらない）になる。
   */
  static Path network(Path directory, long seed, double scale) throws IOException {
    return network(directory, GrpRefitter.shape(GrpFixtures.VARIANT), seed, scale);
  }

  private static Path network(Path directory, GrpShape shape, long seed, double scale) throws IOException {
    Random random = new Random(seed);
    float[] weights = new float[GrpTensor.starts(shape)[GrpTensor.values().length]];
    for (int index = 0; index < weights.length; index++) weights[index] = (float) (random.nextGaussian() * scale);
    GrpCheckpoint.write(GrpNetwork.of(shape, weights), directory);
    return directory.toAbsolutePath().normalize();
  }

  /** 乱数の重みを持つ小さな q（学び直しの GRP と同じフィールド、隠れ層の幅 8）を {@code directory} に書き、正規化した絶対パスを返す。 */
  static Path outcomes(Path directory, Variant variant, long seed) throws IOException {
    return outcomes(directory, new OutcomeShape(variant, GrpRefitter.shape(VARIANT).fields(), 8), seed, 0.3);
  }

  /**
   * 重みの標準偏差を {@code scale} にした q を {@code directory} に書き、正規化した絶対パスを返す。{@code scale} が 0 なら一様な q（終わり方と
   * 打点は一様、立直は 1/2）に、大きければ局の結果を大きく外す q になる。
   */
  static Path outcomes(Path directory, OutcomeShape shape, long seed, double scale) throws IOException {
    Random random = new Random(seed);
    float[] weights = new float[OutcomeTensor.starts(shape)[OutcomeTensor.values().length]];
    for (int index = 0; index < weights.length; index++) weights[index] = (float) (random.nextGaussian() * scale);
    GrpCheckpoint.write(OutcomeNetwork.of(shape, weights), directory);
    return directory.toAbsolutePath().normalize();
  }

  /** 乱数の重みを持つ小さな GruSinkhorn を {@code directory} に書き、正規化した絶対パスを返す。 */
  static Path gru(Path directory, long seed) throws IOException {
    GruShape shape = new GruShape(VARIANT, 8, 1);
    Random random = new Random(seed);
    float[] weights = new float[GruTensor.count(shape)];
    for (int index = 0; index < weights.length; index++) weights[index] = (float) (random.nextGaussian() * 0.3);
    GrpCheckpoint.write(GruSinkhorn.of(shape, weights), directory);
    return directory.toAbsolutePath().normalize();
  }

  /** テストの設定の既定値に「キーのパス=値」の上書きを重ねて、GRP の裏の処理と Φ の決め方が受け取る値を作る。 */
  static PhiSetup setup(String... assignments) {
    PrismSettings settings = TestSettings.of(assignments);
    return new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed());
  }

  /** 自前の GRP の run で、点数の順位から始め、学び直しは手動、切り替えは提案だけにしたもの（上書きを重ねる）。 */
  static PhiSetup scoreStart(String... assignments) {
    return setup(concat(scoreStartAssignments(), assignments));
  }

  /** 自前の GRP の run で、checkpoint の GRP から始め、学び直しは手動、切り替えは提案だけにしたもの（上書きを重ねる）。 */
  static PhiSetup grpStart(Path checkpoint, String... assignments) {
    return setup(concat(grpStartAssignments(checkpoint), assignments));
  }

  /** {@link #scoreStart} の上書き。 */
  static String[] scoreStartAssignments() {
    return new String[] {
      "phi.source=SELF_GRP", "phi.start.kind=SCORE_RANK", "phi.grp.monitor=RECORD", "phi.grp.refit=MANUAL",
      "phi.grp.switching=PROPOSE"
    };
  }

  /** {@link #grpStart} の上書き。 */
  static String[] grpStartAssignments(Path checkpoint) {
    return new String[] {
      "phi.source=SELF_GRP", "phi.start.kind=GRP_CHECKPOINT", "phi.start.grp=" + checkpoint, "phi.grp.monitor=RECORD",
      "phi.grp.refit=MANUAL", "phi.grp.switching=PROPOSE"
    };
  }

  static String[] concat(String[] first, String... second) {
    String[] all = new String[first.length + second.length];
    System.arraycopy(first, 0, all, 0, first.length);
    System.arraycopy(second, 0, all, first.length, second.length);
    return all;
  }

  /** run の設定の 1 行目を、テストの設定の既定値に上書きを重ねて書く。 */
  static void writeSettings(Path run, String... assignments) throws IOException {
    TestSettings.create(run, assignments);
  }

  /** run/state.json に次のコホートだけを書く。 */
  static void writeState(Path run, int next) throws IOException {
    Files.createDirectories(run);
    Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": " + next + "}");
  }

  /** コホート {@code t} の行を確定済みにする。 */
  static void seal(GrpFiles files, int t) throws IOException {
    Files.createDirectories(files.cohort(t));
    Files.writeString(files.cohort(t).resolve(CohortWriter.MANIFEST), "{}");
  }

  /** コホート {@code t} を確定し、項目 {@code entry} で集めた半荘の記録 {@code games} を書く。 */
  static void record(GrpFiles files, int t, PhiEntry entry, List<GameRecord> games) throws IOException {
    seal(files, t);
    Files.createDirectories(files.records(t).getParent());
    GameRecordCodec.write(files.records(t), new GameCohort(VARIANT, t, entry.id(), games));
  }

  /** 時点 {@code slot} の READY の候補の行を refits.jsonl に書く。 */
  static RefitRecord ready(GrpFiles files, String id, int slot, Path checkpoint) throws IOException {
    return ready(files, id, slot, checkpoint, null);
  }

  /** 時点 {@code slot} の READY の候補の行を、q（{@code outcomes}。無ければ null）と一緒に refits.jsonl に書く。 */
  static RefitRecord ready(GrpFiles files, String id, int slot, Path checkpoint, Path outcomes) throws IOException {
    var record =
        new RefitRecord(
            id,
            slot,
            RefitStatus.READY,
            RefitCause.MANUAL,
            new RecencyWindow(slot, 2, 64),
            GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID,
            checkpoint.toString(),
            outcomes == null ? "" : outcomes.toString(),
            "",
            null,
            null,
            "",
            1.0);
    record.append(files);
    return record;
  }

  /** 点数の順位の項目。 */
  static PhiEntry scoreEntry(String id, int from) {
    return new PhiEntry(id, PhiEntryKind.SCORE_RANK, VARIANT, "", "", "", "", "", "", from, "", "test");
  }

  /** 自己対局の候補の項目（checkpoint は読まない）。 */
  static PhiEntry selfEntry(String id, int from, String sha) {
    return new PhiEntry(
        id, PhiEntryKind.NETWORK, VARIANT, "/grp/" + id, sha, "", "", "", "", from, PhiEntry.INITIAL_ID, "test");
  }
}
