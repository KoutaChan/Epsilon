package com.epsilon.classic.model;

import ai.djl.ndarray.NDManager;
import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * classic のチェックポイント（1 つのディレクトリ）を書き、読む。weights.bin にはネットワークのパラメーターを子ブロックの登録順に書き、
 * model.json には {@link ClassicHeader} を書く。ファイルの書き方と読み方は {@link CheckpointFiles} に従う。
 */
public final class ClassicCheckpoint {

  private ClassicCheckpoint() {}

  /**
   * 初期化済みのネットワークを、新しいディレクトリに書く。ディレクトリが既にあれば、何も書かずに IOException を投げる。
   *
   * @param directory 書くディレクトリ
   * @param network 書くネットワーク
   * @param step 学習のステップ数
   */
  public static void write(Path directory, ClassicNetwork network, long step) throws IOException {
    CheckpointFiles.write(directory, network, network.header(step));
  }

  /** model.json のヘッダーを読む。 */
  public static ClassicHeader header(Path directory) throws IOException {
    return CheckpointFiles.readHeader(directory, ClassicHeader.class);
  }

  /**
   * model.json のヘッダーを読み、ルールと入力のフィンガープリントが今のコードと合うかを確かめる。重みは読まない。
   *
   * @param directory チェックポイントのディレクトリ
   * @param variant 使うルール
   * @return 確かめたヘッダー
   * @throws IllegalStateException ルールか入力のフィンガープリントがチェックポイントと合わない
   */
  public static ClassicHeader verify(Path directory, Variant variant) throws IOException {
    Objects.requireNonNull(variant, "variant");
    ClassicHeader header = header(directory);
    if (header.variant() != variant) {
      throw new IllegalStateException("Checkpoint " + directory + " is for " + header.variant() + ", not " + variant);
    }
    String fingerprint = ClassicSchema.of(header.architecture()).fingerprint();
    if (!header.inputFingerprint().equals(fingerprint)) {
      throw new IllegalStateException(
          "Checkpoint " + directory + " has input fingerprint " + header.inputFingerprint() + ", not " + fingerprint);
    }
    return header;
  }

  /**
   * ヘッダーを確かめてからネットワークを組み、重みを読む。
   *
   * @param directory チェックポイントのディレクトリ
   * @param variant 使うルール
   * @param manager パラメーターと定数を持つマネージャー（デバイスもこのマネージャーのもの）
   * @return 読んだネットワーク
   * @throws IllegalStateException ルールか入力のフィンガープリントがチェックポイントと合わない
   */
  public static ClassicNetwork read(Path directory, Variant variant, NDManager manager) throws IOException {
    ClassicHeader header = verify(directory, variant);
    ClassicNetwork network = new ClassicNetwork(header.architecture(), header.variant(), header.utility());
    network.initialize(manager);
    CheckpointFiles.loadParameters(directory, network, manager);
    return network;
  }
}
