package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * 一対局単位 {@link EpsilonDecisionFragment} のバイナリコーデック。
 *
 * <p>終局順位、GRPの局履歴、GRP 4x4分布は対局ヘッダーへ一度だけ保存する。Decision行は境界インデックスと席判断
 * 通し番号で付随情報を参照し、旧形式の行ごとのGRP・終局順位重複は持たない。
 *
 * <p>型付き入力と方策分布は外部データ本体専用で、埋め込み代替処理は持たない。現行形式識別子だけを受理し、 旧学習データ片は明示的に拒否する。
 */
final class EpsilonDecisionFragmentCodec {

  private static final int MAGIC = 0xED10_0023;

  private EpsilonDecisionFragmentCodec() {}

  static void write(OutputStream rawOut, EpsilonDecisionFragment fragment) throws IOException {
    DataOutputStream out = new DataOutputStream(rawOut);
    EpsilonDecisionBinaryArrayCodec.Scratch scratch = new EpsilonDecisionBinaryArrayCodec.Scratch();
    EpsilonDecisionCompletedGame game = fragment.game();
    List<EpsilonDecisionSampleRecord> samples = game.samples();
    out.writeInt(MAGIC);
    out.writeLong(fragment.actorSnapshotId());
    out.writeInt(fragment.grpTeacherIteration());
    out.writeLong(game.gameId());
    out.writeInt(game.finalRanksCode());
    writeTrainingTargetIdentity(out, fragment.trainingTargetIdentity());
    out.writeInt(samples.size());
    writeBoundaries(out, game.boundaries(), scratch);

    FragmentPayloadRefs payloadRefs = collectExternalPayloadRefs(samples);
    writePayloadFileTable(out, payloadRefs.files());
    for (int index = 0; index < samples.size(); index++) {
      writeSample(out, samples.get(index), payloadRefs.refs().get(index), scratch);
    }
    out.flush();
  }

  /** GRP 付随情報・サンプル本体・データ本体を読まず、学習器の選抜に必要なヘッダー先頭部分だけを返す。 */
  static EpsilonDecisionFragmentStore.FragmentMetadata readHeaderMetadata(InputStream rawIn)
      throws IOException {
    DataInputStream in = new DataInputStream(rawIn);
    FragmentPrefix prefix = readPrefix(in, in.readInt());
    return new EpsilonDecisionFragmentStore.FragmentMetadata(
        prefix.sampleCount(),
        prefix.actorSnapshotId(),
        prefix.grpTeacherIteration(),
        prefix.trainingTargetIdentity());
  }

  static EpsilonDecisionFragment read(InputStream rawIn) throws IOException {
    DataInputStream in = new DataInputStream(rawIn);
    EpsilonDecisionBinaryArrayCodec.Scratch scratch = new EpsilonDecisionBinaryArrayCodec.Scratch();
    FragmentHeader header = readHeader(in, in.readInt(), scratch);
    PayloadFileTable payloadFiles = readPayloadFileTable(in);
    ArrayList<EpsilonDecisionSampleRecord> samples = new ArrayList<>(header.sampleCount());
    for (int index = 0; index < header.sampleCount(); index++) {
      samples.add(readSample(in, scratch, payloadFiles, header));
    }
    try {
      return new EpsilonDecisionFragment(
          header.actorSnapshotId(),
          header.grpTeacherIteration(),
          new EpsilonDecisionCompletedGame(
              header.gameId(), header.finalRanksCode(), header.boundaries(), samples),
          header.trainingTargetIdentity());
    } catch (IllegalArgumentException e) {
      throw new IOException("Invalid Decision fragment", e);
    }
  }

  /**
   * 指定全体の位置の軽量記述情報だけを復元する。
   *
   * <p>選ばれなかった行は配列を生成せずに読み飛ばし、選ばれた行も密な入力と方策データ本体は外部ファイルに残す。
   */
  static ArrayList<EpsilonDecisionTrainingSampleDescriptor> readSelected(
      InputStream rawIn,
      BitSet selectedIndexes,
      int globalOffset,
      int expectedSampleCount,
      EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity)
      throws IOException {
    if (globalOffset < 0) {
      throw new IllegalArgumentException("globalOffset must be non-negative");
    }
    DataInputStream in = new DataInputStream(rawIn);
    EpsilonDecisionBinaryArrayCodec.Scratch scratch = new EpsilonDecisionBinaryArrayCodec.Scratch();
    FragmentHeader header = readHeader(in, in.readInt(), scratch);
    requireExpectedIdentity(header, expectedIdentity);
    if (header.sampleCount() != expectedSampleCount) {
      throw new IOException(
          "Decision fragment sample count changed after planning: expected="
              + expectedSampleCount
              + " actual="
              + header.sampleCount());
    }
    PayloadFileTable payloadFiles = readPayloadFileTable(in);
    ArrayList<EpsilonDecisionTrainingSampleDescriptor> selected =
        new ArrayList<>(selectedCount(selectedIndexes, globalOffset, header.sampleCount()));
    for (int localIndex = 0; localIndex < header.sampleCount(); localIndex++) {
      if (selectedIndexes.get(Math.addExact(globalOffset, localIndex))) {
        selected.add(
            EpsilonDecisionTrainingSampleDescriptor.from(
                readSample(in, scratch, payloadFiles, header)));
      } else {
        skipSample(in, payloadFiles);
      }
    }
    return selected;
  }

  private static void requireExpectedIdentity(
      FragmentHeader header, EpsilonDecisionFragmentStore.ExpectedIdentity expected)
      throws IOException {
    if (header.actorSnapshotId() != expected.actorSnapshotId()
        || header.grpTeacherIteration() != expected.grpTeacherIteration()
        || !header.trainingTargetIdentity().equals(expected.trainingTargetIdentity())) {
      throw new IOException(
          "Decision fragment identity changed after planning: expected="
              + expected
              + " actualActor="
              + header.actorSnapshotId()
              + " actualGrpTeacher="
              + header.grpTeacherIteration()
              + " actualTarget="
              + header.trainingTargetIdentity());
    }
  }

  private static FragmentHeader readHeader(
      DataInputStream in, int magic, EpsilonDecisionBinaryArrayCodec.Scratch scratch)
      throws IOException {
    FragmentPrefix prefix = readPrefix(in, magic);
    List<EpsilonDecisionGameBoundary> boundaries = readBoundaries(in, scratch);
    return new FragmentHeader(
        prefix.actorSnapshotId(),
        prefix.grpTeacherIteration(),
        prefix.gameId(),
        prefix.finalRanksCode(),
        prefix.sampleCount(),
        boundaries,
        prefix.trainingTargetIdentity());
  }

  private static FragmentPrefix readPrefix(DataInputStream in, int magic) throws IOException {
    requireCurrentMagic(magic);
    long actorSnapshotId = in.readLong();
    int grpTeacherIteration = in.readInt();
    requireGrpTeacherIteration(grpTeacherIteration);
    long gameId = in.readLong();
    int finalRanksCode = in.readInt();
    if (!EpsilonGrpRanks.isValidCode(finalRanksCode)) {
      throw new IOException("Invalid completed-game final-ranks code: " + finalRanksCode);
    }
    EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity = readTrainingTargetIdentity(in);
    requireGenerationIdentity(actorSnapshotId);
    int sampleCount = readSampleCount(in);
    return new FragmentPrefix(
        actorSnapshotId,
        grpTeacherIteration,
        gameId,
        finalRanksCode,
        sampleCount,
        trainingTargetIdentity);
  }

  private static void writeBoundaries(
      DataOutputStream out,
      List<EpsilonDecisionGameBoundary> boundaries,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch)
      throws IOException {
    out.writeInt(boundaries.size());
    for (EpsilonDecisionGameBoundary boundary : boundaries) {
      out.writeInt(boundary.index());
      EpsilonDecisionBinaryArrayCodec.writeFloatArray(
          out, boundary.grpFeatureSequenceView(), scratch);
      EpsilonDecisionBinaryArrayCodec.writeFloatArray(
          out, boundary.grpRankProbabilitiesView(), scratch);
    }
  }

  private static List<EpsilonDecisionGameBoundary> readBoundaries(
      DataInputStream in, EpsilonDecisionBinaryArrayCodec.Scratch scratch) throws IOException {
    int count = in.readInt();
    if (count < 0) {
      throw new IOException("Negative Decision boundary count: " + count);
    }
    ArrayList<EpsilonDecisionGameBoundary> boundaries = new ArrayList<>(count);
    try {
      for (int index = 0; index < count; index++) {
        boundaries.add(
            EpsilonDecisionGameBoundary.fromOwnedArrays(
                in.readInt(),
                EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch),
                EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch)));
      }
    } catch (IllegalArgumentException e) {
      throw new IOException("Invalid Decision boundary sidecar", e);
    }
    return boundaries;
  }

  private static void requireCurrentMagic(int magic) throws IOException {
    if (magic != MAGIC) {
      throw new UnsupportedFormatException(
          "Unsupported Decision fragment magic: 0x" + Integer.toHexString(magic));
    }
  }

  private static void requireGrpTeacherIteration(int iteration) throws IOException {
    if (iteration < -1) {
      throw new IOException("Invalid GRP teacher iteration: " + iteration);
    }
  }

  private static void requireGenerationIdentity(long actorSnapshotId) throws IOException {
    if (actorSnapshotId <= 0L) {
      throw new IOException("Invalid actor snapshot ID: " + actorSnapshotId);
    }
  }

  private static int readSampleCount(DataInputStream in) throws IOException {
    int sampleCount = in.readInt();
    if (sampleCount < 0) {
      throw new IOException("Negative Decision sample count: " + sampleCount);
    }
    return sampleCount;
  }

  private static FragmentPayloadRefs collectExternalPayloadRefs(
      List<EpsilonDecisionSampleRecord> samples) throws IOException {
    LinkedHashMap<String, Integer> fileIds = new LinkedHashMap<>();
    ArrayList<String> files = new ArrayList<>();
    ArrayList<SamplePayloadRef> refs = new ArrayList<>(samples.size());
    for (int index = 0; index < samples.size(); index++) {
      FilePayloadRef ref = externalPayloadRef(samples.get(index), index);
      Integer fileId = fileIds.get(ref.file());
      if (fileId == null) {
        fileId = files.size();
        fileIds.put(ref.file(), fileId);
        files.add(ref.file());
      }
      refs.add(new SamplePayloadRef(fileId, ref.shard(), ref.offset(), ref.length(), ref.bucket()));
    }
    return new FragmentPayloadRefs(files, refs);
  }

  private static FilePayloadRef externalPayloadRef(EpsilonDecisionSampleRecord sample, int index)
      throws IOException {
    if (!(sample instanceof EpsilonDecisionDeferredSample deferred)) {
      throw new IOException(
          "Decision fragment format is external-payload only; sample "
              + index
              + " is not deferred: "
              + sample.getClass().getName());
    }
    return deferred.payloadStore().externalFileRef(deferred.payloadRef());
  }

  private static void writeSample(
      DataOutputStream out,
      EpsilonDecisionSampleRecord sample,
      SamplePayloadRef payloadRef,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch)
      throws IOException {
    writeSamplePayloadRef(out, payloadRef);
    EpsilonDecisionBinaryArrayCodec.writeIntArray(
        out,
        ((EpsilonDecisionDeferredSample) sample).legalActionIdsView(),
        sample.legalActionCount(),
        scratch);
    out.writeInt(sample.legalActionCount());
    out.writeInt(sample.playerSeat());
    out.writeInt(sample.sourcePlayerRelativeSeat());
    out.writeInt(sample.currentPlayerRelativeSeat());
    out.writeInt(sample.chosenLegalSlot());
    out.writeInt(sample.chosenActionId());
    out.writeFloat(sample.behaviorLogProb());
    out.writeFloat(sample.behaviorProb());
    out.writeFloat(sample.valueTarget());
    out.writeFloat(sample.advantage());
    out.writeInt(sample.boundaryIndex());
    out.writeInt(sample.seatDecisionOrdinal());
    out.writeInt(sample.ruleProfile());
    out.writeByte(sample.learningRole().ordinal());
  }

  private static EpsilonDecisionDeferredSample readSample(
      DataInputStream in,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch,
      PayloadFileTable payloadFiles,
      FragmentHeader header)
      throws IOException {
    FilePayloadRef payloadRef = readSamplePayloadRef(in, payloadFiles);
    int[] legalActionIds = EpsilonDecisionBinaryArrayCodec.readIntArray(in, scratch);
    int legalActionCount = in.readInt();
    int playerSeat = in.readInt();
    int sourcePlayerRelativeSeat = in.readInt();
    int currentPlayerRelativeSeat = in.readInt();
    int chosenLegalSlot = in.readInt();
    int chosenActionId = in.readInt();
    float behaviorLogProb = in.readFloat();
    float behaviorProb = in.readFloat();
    float valueTarget = in.readFloat();
    float advantage = in.readFloat();
    int boundaryIndex = in.readInt();
    int seatDecisionOrdinal = in.readInt();
    int ruleProfile = in.readInt();
    int roleCode = in.readUnsignedByte();
    if (roleCode >= DecisionLearningRole.values().length) {
      throw new IOException("Invalid Decision learning role: " + roleCode);
    }
    if (playerSeat < 0 || playerSeat >= EpsilonGrpRanks.SEAT_COUNT) {
      throw new IOException("Invalid Decision player seat: " + playerSeat);
    }
    if (boundaryIndex < 0 || boundaryIndex >= header.boundaries().size()) {
      throw new IOException("Decision sample boundary index out of range: " + boundaryIndex);
    }
    EpsilonDecisionGameBoundary boundary = header.boundaries().get(boundaryIndex);
    int finalRank = EpsilonGrpRanks.decode(header.finalRanksCode())[playerSeat];

    try {
      return new EpsilonDecisionDeferredSample(
          EpsilonDecisionTrajectoryPayloadStore.externalFileReader(),
          payloadRef,
          legalActionIds,
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
          header.actorSnapshotId(),
          ruleProfile,
          header.gameId(),
          boundaryIndex,
          seatDecisionOrdinal,
          boundary.grpFeatureSequenceView(),
          header.finalRanksCode(),
          DecisionLearningRole.values()[roleCode]);
    } catch (IllegalArgumentException e) {
      throw new IOException("Invalid Decision sample identity", e);
    }
  }

  private static void skipSample(DataInputStream in, PayloadFileTable payloadFiles)
      throws IOException {
    readSamplePayloadRef(in, payloadFiles);
    EpsilonDecisionBinaryArrayCodec.skipIntArray(in);
    in.skipNBytes(6L * Integer.BYTES + 2L * Float.BYTES);
    in.skipNBytes(2L * Float.BYTES);
    in.skipNBytes(3L * Integer.BYTES + Byte.BYTES);
  }

  private static int selectedCount(BitSet selectedIndexes, int globalOffset, int sampleCount) {
    int count = 0;
    int end = Math.addExact(globalOffset, sampleCount);
    for (int selected = selectedIndexes.nextSetBit(globalOffset);
        selected >= 0 && selected < end;
        selected = selectedIndexes.nextSetBit(selected + 1)) {
      count++;
    }
    return count;
  }

  private static void writeTrainingTargetIdentity(
      DataOutputStream out, EpsilonDecisionTrainingTargetIdentity identity) throws IOException {
    out.writeInt(identity.targetProtocolVersion());
    out.writeFloat(identity.causalTraceLambda());
    out.writeFloat(identity.explorationCreditMix());
  }

  private static EpsilonDecisionTrainingTargetIdentity readTrainingTargetIdentity(
      DataInputStream in) throws IOException {
    int targetProtocolVersion = in.readInt();
    float causalTraceLambda = in.readFloat();
    float explorationCreditMix = in.readFloat();
    try {
      return new EpsilonDecisionTrainingTargetIdentity(
          targetProtocolVersion, causalTraceLambda, explorationCreditMix);
    } catch (IllegalArgumentException e) {
      throw new IOException("Invalid Decision training target identity", e);
    }
  }

  private static void writePayloadFileTable(DataOutputStream out, List<String> files)
      throws IOException {
    out.writeInt(files.size());
    for (String file : files) {
      out.writeUTF(file);
    }
  }

  private static PayloadFileTable readPayloadFileTable(DataInputStream in) throws IOException {
    int count = in.readInt();
    if (count < 0) {
      throw new IOException("Negative Decision payload file table size: " + count);
    }
    ArrayList<String> files = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      String file = in.readUTF();
      if (file.isBlank()) {
        throw new IOException("Blank Decision payload file path at table index " + index);
      }
      files.add(file);
    }
    return new PayloadFileTable(files);
  }

  private static void writeSamplePayloadRef(DataOutputStream out, SamplePayloadRef ref)
      throws IOException {
    out.writeInt(ref.fileId());
    out.writeInt(ref.shard());
    out.writeLong(ref.offset());
    out.writeInt(ref.length());
    out.writeInt(ref.bucket().legalActionCapacity());
    out.writeInt(ref.bucket().actionTransitionCapacity());
  }

  private static FilePayloadRef readSamplePayloadRef(DataInputStream in, PayloadFileTable files)
      throws IOException {
    int fileId = in.readInt();
    int shard = in.readInt();
    long offset = in.readLong();
    int length = in.readInt();
    int legalActionCapacity = in.readInt();
    int actionTransitionCapacity = in.readInt();
    if (fileId < 0 || fileId >= files.size()) {
      throw new IOException(
          "Decision payload file id out of range: fileId=" + fileId + " tableSize=" + files.size());
    }
    try {
      DecisionBucket bucket = new DecisionBucket(legalActionCapacity, actionTransitionCapacity);
      return new FilePayloadRef(files.file(fileId), shard, offset, length, bucket);
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid Decision payload reference", invalid);
    }
  }

  private record FragmentHeader(
      long actorSnapshotId,
      int grpTeacherIteration,
      long gameId,
      int finalRanksCode,
      int sampleCount,
      List<EpsilonDecisionGameBoundary> boundaries,
      EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity) {}

  private record FragmentPrefix(
      long actorSnapshotId,
      int grpTeacherIteration,
      long gameId,
      int finalRanksCode,
      int sampleCount,
      EpsilonDecisionTrainingTargetIdentity trainingTargetIdentity) {}

  static final class UnsupportedFormatException extends IOException {
    private UnsupportedFormatException(String message) {
      super(message);
    }
  }

  private record FragmentPayloadRefs(List<String> files, List<SamplePayloadRef> refs) {}

  private record SamplePayloadRef(
      int fileId, int shard, long offset, int length, DecisionBucket bucket) {
    private SamplePayloadRef {
      if (fileId < 0) {
        throw new IllegalArgumentException("fileId must be non-negative");
      }
      if (bucket == null) {
        throw new NullPointerException("bucket");
      }
    }
  }

  private record PayloadFileTable(List<String> files) {
    private int size() {
      return files.size();
    }

    private String file(int id) {
      return files.get(id);
    }
  }
}
