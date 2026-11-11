package com.epsilon.classic.campaign;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.mahjong.rules.Variant;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * 収集の段が確定するコホート（rows/&lt;t&gt;）のファイル。学習器の席の軌跡と、その macro で数えた迷いの分布を書く。どちらも
 * DataOutputStream（ビッグエンディアン）の形で、軌跡は gzip で圧縮する。形式のバージョンは持たない。途中で切れたファイルは読まない（軌跡は
 * gzip の CRC と長さで、迷いの分布はファイルの長さで見分ける）。
 *
 * <pre>
 * trajectories.gz
 *   UTF ルールの名前  i32 半荘の数
 *   半荘ごと: i64 半荘 ID  u16 局の数  u8 GRP の周辺確率があれば 1  （あれば局ごとに f32 × S × S）  i32 × S 終局の点  i32 判断の数
 *     判断ごと: u16 局  u8 選んだ手  f32 振る舞いの確率  u8 選び替えで局の進みが変わりえたら 1  u16 O の long の数  i64 × その数
 * calibration.bin
 *   i32 分布の数  i32 区間の数  （分布ごと、区間ごとに i64 数）
 * </pre>
 */
final class Cohort {

  /** 学習器の席の軌跡のファイル。 */
  static final String TRAJECTORIES = "trajectories.gz";

  /** 迷いの分布のファイル。 */
  static final String CALIBRATION = "calibration.bin";

  private static final int BUFFER = 1 << 16;
  private static final int MAX_UNSIGNED_BYTE = 0xFF;
  private static final int MAX_UNSIGNED_SHORT = 0xFFFF;

  private Cohort() {}

  /**
   * 軌跡と迷いの分布を、確定する前のディレクトリに書く。
   *
   * @return 書いたファイルの大きさの和（バイト）
   */
  static long write(Path directory, Collector.Collected collected) throws IOException {
    Files.createDirectories(directory);
    Path trajectories = directory.resolve(TRAJECTORIES);
    writeTrajectories(trajectories, collected.trajectories());
    Path calibration = directory.resolve(CALIBRATION);
    writeCalibration(calibration, collected.calibration());
    return Files.size(trajectories) + Files.size(calibration);
  }

  /**
   * 確定したコホートの軌跡を読む。
   *
   * @param variant run のルール
   * @throws IOException 読めないか、切れているか、形式に合わないか、ルールが違う
   */
  static TrajectoryBatch trajectories(Path directory, Variant variant) throws IOException {
    Path file = directory.resolve(TRAJECTORIES);
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      Variant written = Variant.valueOf(in.readUTF());
      if (written != variant) {
        throw new IOException("The trajectories " + file + " are for " + written + ", not " + variant);
      }
      int seats = variant.seats();
      int games = in.readInt();
      if (games < 0) {
        throw new IOException("Negative game count in " + file + ": " + games);
      }
      TrajectoryBatch.Collector collector = new TrajectoryBatch.Collector(variant);
      Observation reader = new Observation(variant);
      long[] words = new long[MAX_UNSIGNED_SHORT];
      for (int game = 0; game < games; game++) {
        collector.add(game(in, variant, seats, reader, words, file));
      }
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the last game of " + file);
      }
      return collector.build();
    } catch (EOFException truncated) {
      throw new IOException("Truncated trajectories " + file, truncated);
    } catch (IllegalArgumentException | IllegalStateException | IndexOutOfBoundsException malformed) {
      throw new IOException("Malformed trajectories " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  /**
   * 確定したコホートの迷いの分布を読む。
   *
   * @throws IOException 読めないか、切れているか、形が違う
   */
  static UncertaintyCdf calibration(Path directory) throws IOException {
    Path file = directory.resolve(CALIBRATION);
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file), BUFFER))) {
      int distributions = in.readInt();
      int bins = in.readInt();
      if (distributions < 0 || bins < 0) {
        throw new IOException("Negative shape in " + file + ": " + distributions + " x " + bins);
      }
      long[][] counts = new long[distributions][bins];
      for (long[] distribution : counts) {
        for (int bin = 0; bin < bins; bin++) {
          distribution[bin] = in.readLong();
        }
      }
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the counts of " + file);
      }
      return UncertaintyCdf.of(counts);
    } catch (EOFException truncated) {
      throw new IOException("Truncated calibration " + file, truncated);
    } catch (IllegalArgumentException malformed) {
      throw new IOException("Malformed calibration " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  private static void writeTrajectories(Path file, TrajectoryBatch batch) throws IOException {
    Variant variant = batch.variant();
    Observation reader = new Observation(variant);
    ObservationBatch scratch = new ObservationBatch(variant, MAX_UNSIGNED_SHORT);
    try (DataOutputStream out =
        new DataOutputStream(
            new BufferedOutputStream(new GZIPOutputStream(Files.newOutputStream(file), BUFFER), BUFFER))) {
      out.writeUTF(variant.name());
      out.writeInt(batch.games().size());
      for (GameTrajectory game : batch.games()) {
        out.writeLong(game.id());
        out.writeShort(bounded(game.hands(), MAX_UNSIGNED_SHORT, "hands"));
        out.writeByte(game.hasPriors() ? 1 : 0);
        if (game.hasPriors()) {
          for (int hand = 0; hand < game.hands(); hand++) {
            for (float probability : game.prior(hand)) {
              out.writeFloat(probability);
            }
          }
        }
        for (int score : game.finalScores()) {
          out.writeInt(score);
        }
        out.writeInt(game.size());
        for (int decision = 0; decision < game.size(); decision++) {
          out.writeShort(game.hand(decision));
          out.writeByte(bounded(game.chosen(decision), MAX_UNSIGNED_BYTE, "choice"));
          out.writeFloat(game.behaviour(decision));
          out.writeByte(game.causal(decision) ? 1 : 0);
          scratch.clear();
          scratch.add(game.bind(decision, reader));
          int longs = scratch.longs(0);
          out.writeShort(bounded(longs, MAX_UNSIGNED_SHORT, "observation longs"));
          long[] words = scratch.words();
          int start = scratch.start(0);
          for (int index = 0; index < longs; index++) {
            out.writeLong(words[start + index]);
          }
        }
      }
    }
  }

  private static void writeCalibration(Path file, UncertaintyCdf calibration) throws IOException {
    long[][] counts = calibration.counts();
    try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file), BUFFER))) {
      out.writeInt(counts.length);
      out.writeInt(counts.length == 0 ? 0 : counts[0].length);
      for (long[] distribution : counts) {
        for (long count : distribution) {
          out.writeLong(count);
        }
      }
    }
  }

  /** 半荘 1 つを読んで軌跡に組み直す。局の始まりは、判断の局まで順に足し、最後に残りの局を足す。 */
  private static GameTrajectory game(
      DataInputStream in, Variant variant, int seats, Observation reader, long[] words, Path file) throws IOException {
    long id = in.readLong();
    int hands = in.readUnsignedShort();
    boolean withPriors = in.readUnsignedByte() != 0;
    float[][] priors = withPriors ? new float[hands][seats * seats] : null;
    if (withPriors) {
      for (float[] prior : priors) {
        for (int index = 0; index < prior.length; index++) {
          prior[index] = in.readFloat();
        }
      }
    }
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = in.readInt();
    }
    int decisions = in.readInt();
    if (decisions < 0) {
      throw new IOException("Negative decision count of game " + id + " in " + file);
    }
    GameTrajectory.Builder builder = new GameTrajectory.Builder(variant, id);
    int started = 0;
    for (int decision = 0; decision < decisions; decision++) {
      int hand = in.readUnsignedShort();
      int chosen = in.readUnsignedByte();
      float behaviour = in.readFloat();
      boolean causal = in.readUnsignedByte() != 0;
      int longs = in.readUnsignedShort();
      for (int index = 0; index < longs; index++) {
        words[index] = in.readLong();
      }
      if (hand < started - 1 || hand >= hands) {
        throw new IOException("Decision " + decision + " of game " + id + " in " + file + " is in hand " + hand);
      }
      while (started <= hand) {
        builder.hand(withPriors ? priors[started] : null);
        started++;
      }
      Observation observation = reader.bind(words, 0);
      if (observation.longs() != longs) {
        throw new IOException(
            "Decision " + decision + " of game " + id + " in " + file + " holds " + longs + " longs, its O "
                + observation.longs());
      }
      builder.decision(observation, chosen, behaviour, causal);
    }
    while (started < hands) {
      builder.hand(withPriors ? priors[started] : null);
      started++;
    }
    return builder.finish(scores);
  }

  private static int bounded(int value, int max, String name) {
    if (value < 0 || value > max) {
      throw new IllegalArgumentException(name + " does not fit the cohort: " + value);
    }
    return value;
  }
}
