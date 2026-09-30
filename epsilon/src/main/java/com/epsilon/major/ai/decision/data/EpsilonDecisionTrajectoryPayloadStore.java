package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.io.ZstdFrameBuffer;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionTrainingSlabWriter;
import com.epsilon.major.config.settings.EpsilonSettings;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 対局終了まで教師値が確定しない入力と確率分布を、一時ファイルへ退避する。
 *
 * <p>ヒープにはオフセット・長さとメタデータだけを保持する。一時保存方式では解放時にファイルを削除する。保存継続方式では学習データ片と同じ保存先にファイルを残し、その参照を学習器へ渡す。
 *
 * <p>保存継続方式は、収集終了時に大きな入力や確率分布を再コピーしないために使う。外部ファイルを読み込むのは学習器がサンプル本体を必要とするときだけである。
 */
public interface EpsilonDecisionTrajectoryPayloadStore extends AutoCloseable {

  /** データ本体の所有権だけを渡す参照。ファイル位置や非同期書込みの状態はデータ内に閉じる。 */
  sealed interface PayloadRef permits FilePayloadRef, AsyncFilePayloadRef {}

  static EpsilonDecisionTrajectoryPayloadStore fromSettings(Path baseDir, boolean durable)
      throws IOException {
    return fromSettings(
        baseDir,
        durable,
        EpsilonSettings.defaults().bind(DecisionTrainInFlightSpoolSettings.class));
  }

  static EpsilonDecisionTrajectoryPayloadStore fromSettings(
      Path baseDir, boolean durable, DecisionTrainInFlightSpoolSettings settings)
      throws IOException {
    return new AsyncFileBacked(
        resolveDirectories(baseDir, settings.dirs()),
        settings.asyncQueuePayloads(),
        settings.asyncWriters(),
        durable,
        settings.compressionLevel());
  }

  private static List<Path> resolveDirectories(Path baseDir, String configuredDirs) {
    ArrayList<Path> directories = new ArrayList<>();
    for (String configuredDir : configuredDirs.split(",")) {
      String value = configuredDir.trim();
      if (value.isEmpty()) {
        throw new IllegalArgumentException("in-flight spool dirs contains an empty path");
      }
      Path path = Path.of(value);
      directories.add(path.isAbsolute() ? path : baseDir.resolve(path));
    }
    return directories;
  }

  static EpsilonDecisionTrajectoryPayloadStore asyncFileBacked(
      Path dir, int queuePayloads, int writerCount) throws IOException {
    return asyncFileBacked(dir, queuePayloads, writerCount, false);
  }

  static EpsilonDecisionTrajectoryPayloadStore asyncFileBacked(
      Path dir, int queuePayloads, int writerCount, boolean durable) throws IOException {
    return asyncFileBacked(List.of(dir), queuePayloads, writerCount, durable);
  }

  static EpsilonDecisionTrajectoryPayloadStore asyncFileBacked(
      List<Path> dirs, int queuePayloads, int writerCount, boolean durable) throws IOException {
    return new AsyncFileBacked(
        dirs,
        queuePayloads,
        writerCount,
        durable,
        EpsilonSettings.defaults()
            .bind(DecisionTrainInFlightSpoolSettings.class)
            .compressionLevel());
  }

  static EpsilonDecisionTrajectoryPayloadStore externalFileReader() {
    return ExternalFileReader.INSTANCE;
  }

  /** 学習器ワーカー専用のチャネル/一時バッファ再利用セッションを開く。スレッドセーフではない。 */
  static ExternalFileSession openExternalFileSession() {
    return new ExternalFileSession();
  }

  /** 一行の入力と確率配列を複製せずに引き取る。呼出し後は内容を変更しない。 */
  PayloadRef write(DecisionHostBatch.RowSlice input, float[] behaviorPolicy, float[] rolloutPolicy)
      throws IOException;

  EpsilonDecisionTrajectoryPayload read(PayloadRef ref) throws IOException;

  /** 収集済みデータ本体へ終局教師値と軽量メタデータを結び付け、遅延サンプルとして返す。 */
  default EpsilonDecisionSampleRecord deferredSample(
      PayloadRef payloadRef,
      int[] legalActionIdBySlot,
      int legalActionCount,
      int playerSeat,
      int sourcePlayerRelativeSeat,
      int currentPlayerRelativeSeat,
      int chosenLegalSlot,
      int chosenActionId,
      float behaviorLogProb,
      float behaviorProb,
      float valueTarget,
      float advantage,
      int finalRank,
      long actorSnapshotId,
      int ruleProfile,
      long gameId,
      int boundaryIndex,
      int seatDecisionOrdinal,
      float[] grpFeatureSequence,
      int grpFinalRanksCode,
      DecisionLearningRole learningRole,
      DecisionBranchTarget branchTarget) {
    return new EpsilonDecisionDeferredSample(
        this,
        payloadRef,
        legalActionIdBySlot,
        legalActionCount,
        playerSeat,
        sourcePlayerRelativeSeat,
        currentPlayerRelativeSeat,
        chosenLegalSlot,
        chosenActionId,
        behaviorLogProb,
        behaviorProb,
        valueTarget,
        advantage,
        finalRank,
        actorSnapshotId,
        ruleProfile,
        gameId,
        boundaryIndex,
        seatDecisionOrdinal,
        grpFeatureSequence,
        grpFinalRanksCode,
        learningRole,
        branchTarget);
  }

  /**
   * 学習データ片に保存するファイルに保存したデータ本体参照を返す。
   *
   * <p>学習データ片形式は外部-ファイル専用で埋め込み代替処理を持たない。そのためファイルに保存したデータ本体参照を出せない
   * 保存先はここで直ちに例外を送出する。非同期ファイルに保存した実装では書き込み完了をここで待ち、データ本体がディスク上に存在することを保証してから
   * パス/分割ファイル/オフセット/長さと復号前に使う容量区分を返す。
   */
  default FilePayloadRef externalFileRef(PayloadRef ref) throws IOException {
    throw new IOException("Decision fragment requires file-backed payload ref: " + ref);
  }

  /** この保存先が作成したデータ本体ファイルを書き込み処理順で返す。 */
  default List<Path> payloadFiles() {
    return List.of();
  }

  /** 書き込み処理を閉じ、この保存先が作成したデータ本体ファイルを破棄する。 */
  default void discardPayloadFiles() throws IOException {
    close();
  }

  @Override
  default void close() throws IOException {}

  /**
   * 処理中の対局中の行動履歴データ本体を容量制限付きキュー経由で複数書き込み処理に逃がす保存先。
   *
   * <p>{@link #write} はデータ本体を直接ファイルに書かず、キューへ投入して {@link AsyncFilePayloadRef} を返す。書き込み処理は
   * 分割ファイルごとの一時ファイルを持つため、1 つの {@link FileChannel} と {@code writePosition} を複数スレッドで奪い合わない。これにより
   * 対局実行処理の {@code selectActions()} はファイル符号化/書き込みの完了を待たずに次の一括処理へ進める。
   *
   * <p>キューは必ず上限付きにする。書き込み処理が追いつかない場合は {@code write()} 側で入力抑制され、未書き込みデータ本体が
   * ヒープに無制限に積み上がるのを防ぐ。完了前の参照は {@link #read(PayloadRef)} または {@link #externalFileRef} で必要に
   * なった時点で待つため、学習サンプルの内容は同期保存先と同一になる。
   *
   * <p>保存を継続する方式では書き込み処理のデータ本体ファイルを解放時に削除しない。学習データ片が参照するデータ本体ファイルは {@link
   * EpsilonDecisionFragmentStore#discardAll()} が作成から解放までの管理を持つ。
   */
  final class AsyncFileBacked implements EpsilonDecisionTrajectoryPayloadStore {
    private static final WriteTask END = new WriteTask(null, new CompletableFuture<>());

    private final ArrayBlockingQueue<WriteTask> queue;
    private final ArrayList<PayloadWriter> writers;
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private volatile boolean closed;

    private AsyncFileBacked(
        List<Path> dirs, int queuePayloads, int writerCount, boolean durable, int compressionLevel)
        throws IOException {
      if (dirs.isEmpty()) {
        throw new IllegalArgumentException("dirs must not be empty");
      }
      if (writerCount <= 0) {
        throw new IllegalArgumentException("writerCount must be positive");
      }
      this.queue = new ArrayBlockingQueue<>(queuePayloads);
      this.writers = new ArrayList<>(writerCount);
      for (int i = 0; i < writerCount; i++) {
        PayloadWriter writer =
            new PayloadWriter(dirs.get(i % dirs.size()), i, !durable, compressionLevel);
        writers.add(writer);
        writer.start();
      }
    }

    @Override
    public List<Path> payloadFiles() {
      return writers.stream().map(PayloadWriter::file).toList();
    }

    @Override
    public void discardPayloadFiles() throws IOException {
      IOException failureToThrow = null;
      try {
        close();
      } catch (IOException failure) {
        failureToThrow = failure;
      }
      for (PayloadWriter writer : writers) {
        try {
          writer.deleteFile();
        } catch (IOException failure) {
          if (failureToThrow == null) {
            failureToThrow = failure;
          } else {
            failureToThrow.addSuppressed(failure);
          }
        }
      }
      if (failureToThrow != null) {
        throw failureToThrow;
      }
    }

    @Override
    public PayloadRef write(
        DecisionHostBatch.RowSlice input, float[] behaviorPolicy, float[] rolloutPolicy)
        throws IOException {
      EpsilonDecisionTrajectoryPayload payload =
          new EpsilonDecisionTrajectoryPayload(input, behaviorPolicy, rolloutPolicy);
      ensureOpen();
      rethrowFailure();
      CompletableFuture<FilePayloadRef> future = new CompletableFuture<>();
      try {
        while (true) {
          WriteTask task = new WriteTask(payload, future);
          if (queue.offer(task, 100L, TimeUnit.MILLISECONDS)) {
            break;
          }
          ensureOpen();
          rethrowFailure();
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while enqueueing Decision in-flight payload", e);
      }
      rethrowFailure();
      return new AsyncFilePayloadRef(future);
    }

    @Override
    public EpsilonDecisionTrajectoryPayload read(PayloadRef ref) throws IOException {
      FilePayloadRef fileRef = resolveFileRef(ref);
      return writerFor(fileRef).read(fileRef);
    }

    @Override
    public FilePayloadRef externalFileRef(PayloadRef ref) throws IOException {
      return resolveFileRef(ref);
    }

    @Override
    public void close() throws IOException {
      if (closed) {
        return;
      }
      closed = true;
      IOException failureToThrow = null;
      try {
        for (int i = 0; i < writers.size(); i++) {
          enqueueEndMarker();
        }
        for (PayloadWriter writer : writers) {
          writer.join();
        }
        rethrowFailure();
      } catch (IOException e) {
        failureToThrow = e;
      } finally {
        for (PayloadWriter writer : writers) {
          try {
            writer.closeFile();
          } catch (IOException e) {
            if (failureToThrow == null) {
              failureToThrow = e;
            } else {
              failureToThrow.addSuppressed(e);
            }
          }
        }
      }
      if (failureToThrow != null) {
        throw failureToThrow;
      }
    }

    private void enqueueEndMarker() throws IOException {
      try {
        while (!queue.offer(END, 100L, TimeUnit.MILLISECONDS)) {
          rethrowFailure();
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while draining Decision in-flight payload writers", e);
      }
    }

    private FilePayloadRef resolveFileRef(PayloadRef ref) throws IOException {
      if (ref instanceof FilePayloadRef fileRef) {
        return fileRef;
      }
      if (!(ref instanceof AsyncFilePayloadRef asyncRef)) {
        throw new IllegalArgumentException("Expected async file trajectory payload ref: " + ref);
      }
      try {
        return asyncRef.future().get();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while awaiting Decision in-flight payload", e);
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof IOException io) {
          throw io;
        }
        if (cause instanceof RuntimeException runtime) {
          throw runtime;
        }
        throw new IOException("Failed to write Decision in-flight payload", cause);
      }
    }

    private PayloadWriter writerFor(FilePayloadRef fileRef) {
      int shard = fileRef.shard();
      if (shard < 0 || shard >= writers.size()) {
        throw new IllegalArgumentException(
            "Decision trajectory payload ref does not belong to this store: " + fileRef);
      }
      PayloadWriter writer = writers.get(shard);
      if (!writer.accepts(fileRef)) {
        throw new IllegalArgumentException(
            "Decision trajectory payload ref does not belong to this store: " + fileRef);
      }
      return writer;
    }

    private void ensureOpen() throws IOException {
      if (closed) {
        throw new IOException("Decision in-flight payload spool is closed");
      }
    }

    private void rethrowFailure() throws IOException {
      Throwable t = failure.get();
      if (t == null) {
        return;
      }
      if (t instanceof IOException io) {
        throw io;
      }
      if (t instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (t instanceof Error error) {
        throw error;
      }
      throw new IOException("Decision in-flight payload writer failed", t);
    }

    private void recordFailure(Throwable t) {
      if (!failure.compareAndSet(null, t)) {
        return;
      }
      WriteTask queued;
      while ((queued = queue.poll()) != null) {
        if (queued != END) {
          queued.future().completeExceptionally(t);
        }
      }
      for (PayloadWriter writer : writers) {
        writer.interrupt();
      }
    }

    private final class PayloadWriter implements Runnable {
      private final PayloadFile payloadFile;
      private final EpsilonDecisionBinaryArrayCodec.Scratch writeScratch =
          new EpsilonDecisionBinaryArrayCodec.Scratch();
      private final EpsilonDecisionBinaryArrayCodec.Scratch readScratch =
          new EpsilonDecisionBinaryArrayCodec.Scratch();
      private final PayloadCompressionScratch writeCompressionScratch;
      private final PayloadCompressionScratch readCompressionScratch =
          new PayloadCompressionScratch();
      private final Thread thread;

      private PayloadWriter(Path dir, int shard, boolean deleteOnClose, int compressionLevel)
          throws IOException {
        writeCompressionScratch = new PayloadCompressionScratch(compressionLevel);
        this.payloadFile = new PayloadFile(dir, shard, deleteOnClose);
        this.thread =
            new Thread(this, "decision-inflight-payload-writer-" + String.format("%02d", shard));
        this.thread.setDaemon(true);
      }

      private void start() {
        thread.start();
      }

      private void join() throws IOException {
        try {
          thread.join();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new IOException("Interrupted while joining Decision in-flight payload writer", e);
        }
      }

      private void interrupt() {
        thread.interrupt();
      }

      private boolean accepts(FilePayloadRef ref) {
        return payloadFile.accepts(ref);
      }

      private EpsilonDecisionTrajectoryPayload read(FilePayloadRef ref) throws IOException {
        return payloadFile.read(ref, readScratch, readCompressionScratch);
      }

      private void closeFile() throws IOException {
        writeCompressionScratch.close();
        readCompressionScratch.close();
        payloadFile.close();
      }

      private Path file() {
        return payloadFile.file();
      }

      private void deleteFile() throws IOException {
        payloadFile.delete();
      }

      @Override
      public void run() {
        try {
          while (true) {
            WriteTask task = queue.take();
            if (task == END) {
              return;
            }
            try {
              FilePayloadRef ref =
                  payloadFile.write(task.payload(), writeScratch, writeCompressionScratch);
              task.future().complete(ref);
            } catch (Throwable t) {
              task.future().completeExceptionally(t);
              recordFailure(t);
              return;
            }
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          if (!closed && failure.get() == null) {
            recordFailure(e);
          }
        }
      }
    }

    private record WriteTask(
        EpsilonDecisionTrajectoryPayload payload, CompletableFuture<FilePayloadRef> future) {}
  }

  final class PayloadFile implements AutoCloseable {
    private final Path file;
    private final String filePath;
    private final int shard;
    private final FileChannel channel;
    private final boolean deleteOnClose;
    private long writePosition;
    private volatile boolean closed;

    private PayloadFile(Path dir, int shard, boolean deleteOnClose) throws IOException {
      this.shard = shard;
      this.deleteOnClose = deleteOnClose;
      Files.createDirectories(dir);
      this.file =
          Files.createTempFile(
              dir, "decision-inflight-" + String.format("%02d", shard) + "-", ".bin");
      this.filePath = file.toAbsolutePath().normalize().toString();
      this.channel =
          FileChannel.open(
              file,
              StandardOpenOption.CREATE,
              StandardOpenOption.READ,
              StandardOpenOption.WRITE,
              StandardOpenOption.TRUNCATE_EXISTING);
    }

    private synchronized FilePayloadRef write(
        EpsilonDecisionTrajectoryPayload payload,
        EpsilonDecisionBinaryArrayCodec.Scratch writeScratch,
        PayloadCompressionScratch compressionScratch)
        throws IOException {
      ensureOpen();
      long offset = writePosition;
      ByteBuffer compressed = compressionScratch.encode(payload, writeScratch);
      int compressedLength = compressed.remaining();
      writeFully(channel, offset, compressed);
      writePosition = offset + compressedLength;
      return new FilePayloadRef(
          filePath, shard, offset, compressedLength, payload.input().bucket());
    }

    private EpsilonDecisionTrajectoryPayload read(
        FilePayloadRef ref,
        EpsilonDecisionBinaryArrayCodec.Scratch readScratch,
        PayloadCompressionScratch compressionScratch)
        throws IOException {
      ensureOpen();

      return compressionScratch.decode(channel, ref, readScratch);
    }

    @Override
    public synchronized void close() throws IOException {
      if (closed) {
        return;
      }
      closed = true;
      IOException failure = null;
      try {
        channel.close();
      } catch (IOException e) {
        failure = e;
      }
      if (deleteOnClose) {
        try {
          Files.deleteIfExists(file);
        } catch (IOException e) {
          if (failure == null) {
            failure = e;
          } else {
            failure.addSuppressed(e);
          }
        }
      }
      if (failure != null) {
        throw failure;
      }
    }

    private boolean accepts(FilePayloadRef ref) {
      return shard == ref.shard() && filePath.equals(ref.file());
    }

    private Path file() {
      return file;
    }

    private void delete() throws IOException {
      Files.deleteIfExists(file);
    }

    private void ensureOpen() throws IOException {
      if (closed) {
        throw new IOException("Decision in-flight payload spool is closed: " + file);
      }
    }
  }

  /**
   * 学習データ片に保存された外部データ本体参照を読むための状態を持たない保存先。
   *
   * <p>収集用保存先は解放済みでも、学習データ片にはファイルパス / オフセット / 長さが残る。 学習器側はこの
   * 読み取り処理でデータ本体ファイルを直接開き、必要になったサンプルだけデータを復元する。
   */
  final class ExternalFileReader implements EpsilonDecisionTrajectoryPayloadStore {
    private static final ExternalFileReader INSTANCE = new ExternalFileReader();
    private final ArrayDeque<PayloadReadScratch> availableScratch = new ArrayDeque<>();

    private ExternalFileReader() {}

    @Override
    public PayloadRef write(
        DecisionHostBatch.RowSlice input, float[] behaviorPolicy, float[] rolloutPolicy)
        throws IOException {
      throw new IOException("External file payload reader is read-only");
    }

    @Override
    public EpsilonDecisionTrajectoryPayload read(PayloadRef ref) throws IOException {
      if (!(ref instanceof FilePayloadRef fileRef)) {
        throw new IllegalArgumentException("Expected external file payload ref: " + ref);
      }
      PayloadReadScratch scratch;
      synchronized (availableScratch) {
        scratch = availableScratch.pollFirst();
      }
      if (scratch == null) {
        scratch = new PayloadReadScratch();
      }
      try (FileChannel channel =
          FileChannel.open(Path.of(fileRef.file()), StandardOpenOption.READ)) {
        return scratch.compression().decode(channel, fileRef, scratch.arrays());
      } finally {
        synchronized (availableScratch) {
          availableScratch.addFirst(scratch);
        }
      }
    }
  }

  /**
   * 一つのバッチ構築ワーカーが専有する外部データ本体読み取り処理。
   *
   * <p>乱数並べ替え後も同じ少数分割ファイルを繰り返し読むため、サンプルごとのFileChannel 開く/解放を行わない。
   */
  final class ExternalFileSession implements AutoCloseable {
    private final Map<String, FileChannel> channels = new LinkedHashMap<>();
    private final PayloadReadScratch scratch = new PayloadReadScratch();
    private boolean closed;

    /** 外部データ本体をサンプルへ復元せず、呼出し側が所有する最終ホスト側の連続バッファへ直接復号する。 */
    public void readTrainingRow(
        EpsilonDecisionTrainingSampleDescriptor descriptor,
        DecisionTrainingSlabWriter destination,
        int row,
        float actorWeight,
        float sampleWeight)
        throws IOException {
      read(
          descriptor.payloadRef(),
          (in, arrays) -> {
            EpsilonDecisionTrajectoryPayloadCodec.readTrainingRow(
                in, arrays, descriptor, destination, row, actorWeight, sampleWeight);
            return null;
          });
    }

    <T> T read(FilePayloadRef ref, DecodedPayloadReader<T> reader) throws IOException {
      if (closed) {
        throw new IllegalStateException("Decision payload session is closed");
      }

      FileChannel channel = channels.get(ref.file());
      if (channel == null) {
        channel = FileChannel.open(Path.of(ref.file()), StandardOpenOption.READ);
        channels.put(ref.file(), channel);
      }
      return scratch.compression().decodeWith(channel, ref, scratch.arrays(), reader);
    }

    @Override
    public void close() throws IOException {
      if (closed) {
        return;
      }
      closed = true;
      IOException failure = null;
      for (FileChannel channel : channels.values()) {
        try {
          channel.close();
        } catch (IOException closeFailure) {
          if (failure == null) {
            failure = closeFailure;
          } else {
            failure.addSuppressed(closeFailure);
          }
        }
      }
      channels.clear();
      scratch.compression().close();
      if (failure != null) {
        throw failure;
      }
    }
  }

  @FunctionalInterface
  interface DecodedPayloadReader<T> {
    T read(DataInputStream in, EpsilonDecisionBinaryArrayCodec.Scratch scratch) throws IOException;
  }

  final class PayloadCompressionScratch implements AutoCloseable {
    private final ZstdFrameBuffer frame;

    PayloadCompressionScratch() {
      this(0);
    }

    PayloadCompressionScratch(int compressionLevel) {
      frame = new ZstdFrameBuffer(compressionLevel);
    }

    private ByteBuffer encode(
        EpsilonDecisionTrajectoryPayload payload,
        EpsilonDecisionBinaryArrayCodec.Scratch arrayScratch)
        throws IOException {
      DataOutputStream output = frame.resetOutput();
      EpsilonDecisionTrajectoryPayloadCodec.write(output, payload, arrayScratch);
      output.flush();
      return frame.compress();
    }

    private EpsilonDecisionTrajectoryPayload decode(
        FileChannel channel,
        FilePayloadRef ref,
        EpsilonDecisionBinaryArrayCodec.Scratch arrayScratch)
        throws IOException {
      EpsilonDecisionTrajectoryPayload payload =
          decodeWith(
              channel,
              ref,
              arrayScratch,
              (in, scratch) -> EpsilonDecisionTrajectoryPayloadCodec.read(in, scratch));
      if (!payload.input().bucket().equals(ref.bucket())) {
        throw new IOException(
            "Decision trajectory payload bucket mismatch: expected="
                + ref.bucket()
                + " actual="
                + payload.input().bucket());
      }
      return payload;
    }

    private <T> T decodeWith(
        FileChannel channel,
        FilePayloadRef ref,
        EpsilonDecisionBinaryArrayCodec.Scratch arrayScratch,
        DecodedPayloadReader<T> reader)
        throws IOException {
      try (DataInputStream in = frame.read(channel, ref.offset(), ref.length())) {
        T decoded = reader.read(in, arrayScratch);
        if (in.read() != -1) {
          throw new IOException("Decision trajectory payload has trailing bytes");
        }
        return decoded;
      }
    }

    @Override
    public void close() {
      frame.close();
    }
  }

  record PayloadReadScratch(
      EpsilonDecisionBinaryArrayCodec.Scratch arrays, PayloadCompressionScratch compression) {
    private PayloadReadScratch() {
      this(new EpsilonDecisionBinaryArrayCodec.Scratch(), new PayloadCompressionScratch());
    }
  }

  private static void writeFully(FileChannel channel, long position, ByteBuffer buffer)
      throws IOException {
    while (buffer.hasRemaining()) {
      int written = channel.write(buffer, position);
      if (written <= 0) {
        throw new IOException("Failed to write Decision in-flight payload");
      }
      position += written;
    }
  }
}

record EpsilonDecisionTrajectoryPayload(
    com.epsilon.major.ai.decision.input.DecisionHostBatch.RowSlice input,
    float[] behaviorPolicy,
    float[] rolloutPolicy) {

  EpsilonDecisionTrajectoryPayload {
    if (input.size() != 1 || input.hasTrainingTargets()) {
      throw new IllegalArgumentException("trajectory payload input must be one inference row");
    }
    // 内部データ本体は生成後に変更しない前提で一時保存領域/保存先へ渡す。ここで複製すると
    // 1 判断ごとの方策が余分に複製され、処理中の一時保存領域の効果を相殺する。
    behaviorPolicy =
        EpsilonDecisionSample.requireNormalizedSlotPolicy(
            behaviorPolicy,
            input.source().legalActionCount(input.fromInclusive()),
            "behaviorPolicy");
    rolloutPolicy =
        EpsilonDecisionSample.requireNormalizedSlotPolicy(
            rolloutPolicy, input.source().legalActionCount(input.fromInclusive()), "rolloutPolicy");
  }
}

record FilePayloadRef(String file, int shard, long offset, int length, DecisionBucket bucket)
    implements EpsilonDecisionTrajectoryPayloadStore.PayloadRef {
  FilePayloadRef {
    if (file.isBlank()) {
      throw new IllegalArgumentException("file must not be blank");
    }
    if (shard < 0) {
      throw new IllegalArgumentException("shard must be non-negative");
    }
    if (offset < 0L) {
      throw new IllegalArgumentException("offset must be non-negative");
    }
    if (length <= 0) {
      throw new IllegalArgumentException("length must be positive");
    }
    if (bucket == null) {
      throw new NullPointerException("bucket");
    }
  }
}

record AsyncFilePayloadRef(CompletableFuture<FilePayloadRef> future)
    implements EpsilonDecisionTrajectoryPayloadStore.PayloadRef {}
