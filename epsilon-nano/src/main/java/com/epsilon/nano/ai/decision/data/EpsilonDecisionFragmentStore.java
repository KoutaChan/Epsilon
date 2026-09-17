package com.epsilon.nano.ai.decision.data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

/**
 * 反復学習ごとのディレクトリに Decision の学習データ片を一時保存し、学習後に削除する。
 *
 * <p>学習データファイル本体は {@code *.decision-fragment.bin} として保存する。局所的な世代 は反復回数単位の専用ディレクトリを使い、 学習完了後に {@link
 * #discardAll()} でデータ本体と学習データファイルをまとめて破棄する。
 */
public final class EpsilonDecisionFragmentStore {

  private static final String EXTENSION = ".decision-fragment.bin";
  private static final String FILE_PREFIX = "fragment-";
  private static final AtomicLong PUBLISH_SEQUENCE = new AtomicLong();

  private final Path spoolDir;

  /**
   * 指定ディレクトリを学習データファイル一時保存領域として使う保存先を構築する。
   *
   * @param spoolDir 学習データファイルと 遅延読み込みするデータ本体を置く反復回数専用ディレクトリ
   */
  public EpsilonDecisionFragmentStore(Path spoolDir) {
    this.spoolDir = spoolDir;
  }

  /**
   * 公開時の衝突回避用の接尾辞を除き、ファイルの内容に対応する固定の乱数補助値を求める。
   *
   * <p>本番形式以外のファイル名では従来のパスのハッシュ値を使い、テストや明示入力元の順序を変えない。
   */
  public static int stableShuffleSalt(Path path) {
    Objects.requireNonNull(path, "path");
    String filename = path.getFileName().toString();
    if (!filename.startsWith(FILE_PREFIX) || !filename.endsWith(EXTENSION)) {
      return path.hashCode();
    }
    String stem = filename.substring(0, filename.length() - EXTENSION.length());
    int suffixSeparator = stem.lastIndexOf('-');
    if (suffixSeparator <= FILE_PREFIX.length()
        || suffixSeparator + 1 >= stem.length()
        || !stem.substring(suffixSeparator + 1).chars().allMatch(Character::isDigit)) {
      return path.hashCode();
    }
    return stem.substring(0, suffixSeparator).hashCode();
  }

  /**
   * 学習データファイルを一時ファイルへ書き、完成後に準備済み名へ不可分な操作で公開する。
   *
   * @param fragment 永続化する本番学習データファイル
   * @return 公開済み準備済みファイル
   * @throws IOException ディレクトリ作成、書込、または名前変更に失敗した場合
   */
  public Path publish(EpsilonDecisionFragment fragment) throws IOException {
    Files.createDirectories(spoolDir);
    String name =
        FILE_PREFIX
            + fragment.actorSnapshotId()
            + "-"
            + fragment.game().gameId()
            + "-"
            + PUBLISH_SEQUENCE.getAndIncrement()
            + EXTENSION;
    Path tmp = spoolDir.resolve(name + ".tmp");
    Path ready = spoolDir.resolve(name);
    try {
      try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(tmp), 1 << 16)) {
        EpsilonDecisionFragmentCodec.write(out, fragment);
      }
      moveReady(tmp, ready);
      return ready;
    } finally {
      Files.deleteIfExists(tmp);
    }
  }

  /**
   * 単一学習データファイルをデコードして返す。
   *
   * @param path 読み込む学習データファイル
   * @return ヘッダーとサンプルレコードを含む学習データファイル
   * @throws IOException ファイルの読込または復号に失敗した場合
   */
  public EpsilonDecisionFragment loadFragment(Path path) throws IOException {
    try (InputStream in = new BufferedInputStream(Files.newInputStream(path), 1 << 16)) {
      return EpsilonDecisionFragmentCodec.read(in);
    }
  }

  /** 外部データ本体を開かず、選抜済み行の軽量記述情報だけを復元する。 */
  public ArrayList<EpsilonDecisionTrainingSampleDescriptor> loadSelectedSamples(
      Path path,
      BitSet selectedIndexes,
      int globalOffset,
      int expectedSampleCount,
      ExpectedIdentity expectedIdentity)
      throws IOException {
    try (InputStream in = new BufferedInputStream(Files.newInputStream(path), 1 << 16)) {
      return EpsilonDecisionFragmentCodec.readSelected(
          in, selectedIndexes, globalOffset, expectedSampleCount, expectedIdentity);
    }
  }

  /**
   * データ本体を読まずに対局数と方策モデル・GRP・教師値識別情報を返す。
   *
   * @param path 読み込む学習データファイル
   * @return ヘッダーだけから復元したメタデータ
   * @throws IOException ファイルの読込またはヘッダー復号に失敗した場合
   */
  public FragmentMetadata fragmentMetadata(Path path) throws IOException {
    try (InputStream in = new BufferedInputStream(Files.newInputStream(path), 1 << 16)) {
      return EpsilonDecisionFragmentCodec.readHeaderMetadata(in);
    }
  }

  /**
   * 学習データファイルヘッダーが現在学習の反復で固定した方策モデル/GRP/教師値識別情報と完全一致することを検証する。
   *
   * <p>サンプルデータ本体を読む前の第1段階と、データ本体を復元する直前の第2段階の両方から呼ぶ。
   *
   * @param path 不一致時の診断に含める学習データファイルパス
   * @param actual 学習データファイルヘッダーから読んだ識別情報
   * @param expected 現在学習の反復が固定した識別情報
   * @throws IOException 一項目でも一致しない場合
   */
  public static void requireExactIdentity(
      Path path, FragmentMetadata actual, ExpectedIdentity expected) throws IOException {
    Objects.requireNonNull(expected, "expected");
    if (actual.actorSnapshotId() != expected.actorSnapshotId()
        || actual.grpTeacherIteration() != expected.grpTeacherIteration()
        || !actual.trainingTargetIdentity().equals(expected.trainingTargetIdentity())) {
      throw new IOException(
          "Decision fragment identity mismatch before payload materialization: path="
              + path
              + " expected="
              + expected
              + " actual="
              + actual);
    }
  }

  /**
   * 自己対局反復回数用の一時一時保存領域を丸ごと削除する。
   *
   * @throws IOException ファイルまたはディレクトリの削除に失敗した場合
   */
  public void discardAll() throws IOException {
    discardSpool(spoolDir);
  }

  /** 保存先を構築せず、失敗した収集の一時保存領域をディレクトリごと破棄する。 */
  public static void discardSpool(Path dir) throws IOException {
    if (!Files.exists(dir)) {
      return;
    }
    try (Stream<Path> stream = Files.walk(dir)) {
      for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }

  private static void moveReady(Path tmp, Path ready) throws IOException {
    try {
      Files.move(tmp, ready, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } catch (AtomicMoveNotSupportedException e) {
      Files.move(tmp, ready, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  /**
   * データ本体を展開せずに読める学習データファイルヘッダー。
   *
   * @param samples 学習データファイルに含まれるDecision行数
   * @param actorSnapshotId 学習の反復開始時候補のスナップショット ID
   * @param grpTeacherIteration 固定 GRP 教師モデルの反復回数
   * @param trainingTargetIdentity 学習データファイルが従う教師値規約識別情報
   */
  public record FragmentMetadata(
      int samples,
      long actorSnapshotId,
      int grpTeacherIteration,
      EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity) {}

  /**
   * 現在学習の反復で唯一受理する学習データファイルヘッダー識別情報。
   *
   * @param actorSnapshotId 期待する学習の反復開始時スナップショット ID
   * @param grpTeacherIteration 期待する固定 GRP 教師モデル反復回数
   * @param trainingTargetIdentity 期待する教師値規約識別情報
   */
  public record ExpectedIdentity(
      long actorSnapshotId,
      int grpTeacherIteration,
      EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity) {}
}
