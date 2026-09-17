package com.epsilon.major.ai.decision.training;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.Objects;
import java.util.regex.Pattern;

/** 対局用・本番用の採用モデルへの参照と、本番採用の監査記録を管理する。 */
final class EpsilonDecisionChampionStore {

  private static final String POINTER_FILE = "current.json";
  private static final int POINTER_VERSION = 3;
  private static final int PRODUCTION_AUDIT_VERSION = 1;
  private static final Pattern AUDIT_ID = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,127}");
  private static final Gson GSON = new GsonBuilder().serializeNulls().setPrettyPrinting().create();

  private EpsilonDecisionChampionStore() {}

  /** 評価で採用された次のチェックポイントへ対局用の参照を更新する。 */
  static synchronized void promoteArenaChampion(
      Path checkpointRoot, Path candidate, String evidenceRef) throws IOException {
    requireNonBlank(evidenceRef, "evidenceRef");
    Path root = normalizeRoot(checkpointRoot);
    Path normalized = normalizeCandidate(root, candidate);
    withChampionLock(
        root,
        () -> {
          Path previous = resolveArenaChampionStrict(root);
          if (previous == null) {
            throw new IOException("Arena champion is not initialized: " + root);
          }
          if (normalized.equals(previous)) {
            return null;
          }
          int iteration = EpsilonDecisionCheckpointManager.loadManifest(normalized).iteration;
          int previousIteration = EpsilonDecisionCheckpointManager.loadManifest(previous).iteration;
          if ((long) iteration != (long) previousIteration + 1L) {
            throw new IOException(
                "Arena candidate is not the direct champion successor: champion="
                    + previousIteration
                    + " candidate="
                    + iteration);
          }
          publishChampionPointer(
              root,
              arenaChampionPointer(root),
              ChampionRole.ARENA,
              normalized,
              evidenceRef,
              null,
              null,
              null);
          return null;
        });
  }

  /** 独立監査済みの対局用採用モデルへ本番参照を更新する。 */
  static synchronized void promoteProductionChampion(
      Path checkpointRoot,
      Path candidate,
      String auditRef,
      String parentCandidateId,
      String auditId,
      String auditProtocolId)
      throws IOException {
    requireNonBlank(auditRef, "auditRef");
    requireNonBlank(parentCandidateId, "parentCandidateId");
    validateAuditId(auditId);
    requireNonBlank(auditProtocolId, "auditProtocolId");
    Path root = normalizeRoot(checkpointRoot);
    Path normalized = normalizeCandidate(root, candidate);
    withChampionLock(
        root,
        () -> {
          EpsilonDecisionCheckpointBundle current =
              EpsilonDecisionCheckpointManager.loadManifest(normalized);
          Path arena = resolveArenaChampionStrict(root);
          if (!normalized.equals(arena)) {
            throw new IOException(
                "Production promotion requires the current arena champion: candidate="
                    + normalized
                    + " arena="
                    + arena);
          }
          Path previous = resolveProductionChampionStrict(root);
          if (previous == null) {
            throw new IOException("Production champion is not initialized: " + root);
          }
          Path pointerPath = productionChampionPointer(root);
          if (normalized.equals(previous)) {
            ChampionPointer pointer = loadChampionPointer(pointerPath);
            if (!Objects.equals(parentCandidateId, pointer.parentCandidateId())
                || !Objects.equals(auditId, pointer.auditId())
                || !Objects.equals(auditProtocolId, pointer.auditProtocolId())) {
              throw new IOException(
                  "Production champion was promoted by a different independent audit");
            }
            requireProductionAudit(root, pointer);
            return null;
          }
          EpsilonDecisionCheckpointBundle previousManifest =
              EpsilonDecisionCheckpointManager.loadManifest(previous);
          if (!EpsilonDecisionCheckpointManager.candidateId(previous).equals(parentCandidateId)) {
            throw new IOException(
                "Independent audit parent differs from the current production champion");
          }
          if (current.iteration <= previousManifest.iteration) {
            throw new IOException(
                "Refusing to regress production champion: current="
                    + previousManifest.iteration
                    + " candidate="
                    + current.iteration);
          }
          ProductionAudit audit =
              new ProductionAudit(
                  PRODUCTION_AUDIT_VERSION,
                  auditId,
                  auditProtocolId,
                  EpsilonDecisionCheckpointManager.candidateId(normalized),
                  parentCandidateId,
                  auditRef,
                  Instant.now().toString());
          publishProductionAudit(root, audit);
          publishChampionPointer(
              root,
              pointerPath,
              ChampionRole.PRODUCTION,
              normalized,
              auditRef,
              parentCandidateId,
              auditId,
              auditProtocolId);
          return null;
        });
  }

  static synchronized Path resolveArenaChampionStrict(Path checkpointRoot) throws IOException {
    Path root = normalizeRoot(checkpointRoot);
    return resolveChampionPointer(root, arenaChampionPointer(root), ChampionRole.ARENA);
  }

  static synchronized Path resolveProductionChampionStrict(Path checkpointRoot) throws IOException {
    Path root = normalizeRoot(checkpointRoot);
    return resolveChampionPointer(root, productionChampionPointer(root), ChampionRole.PRODUCTION);
  }

  static Path arenaChampionPointer(Path checkpointRoot) throws IOException {
    return normalizeRoot(checkpointRoot).resolve("arena").resolve(POINTER_FILE);
  }

  static Path productionChampionPointer(Path checkpointRoot) throws IOException {
    return normalizeRoot(checkpointRoot).resolve("production").resolve(POINTER_FILE);
  }

  private static Path resolveChampionPointer(Path root, Path pointerPath, ChampionRole role)
      throws IOException {
    if (!Files.exists(pointerPath)) {
      return null;
    }
    ChampionPointer pointer = loadChampionPointer(pointerPath);
    if (pointer.role() != role) {
      throw new IOException("Decision champion pointer role mismatch: " + pointerPath);
    }
    Path candidate = resolveRelativeCandidate(root, pointer.relativeCheckpoint(), role.name());
    String candidateId = EpsilonDecisionCheckpointManager.candidateId(candidate);
    int iteration = EpsilonDecisionCheckpointManager.loadManifest(candidate).iteration;
    if (!pointer.candidateId().equals(candidateId) || pointer.iteration() != iteration) {
      throw new IOException("Decision champion pointer does not match checkpoint: " + candidate);
    }
    if (role == ChampionRole.PRODUCTION && pointer.auditId() != null) {
      requireProductionAudit(root, pointer);
    }
    return candidate;
  }

  private static void publishChampionPointer(
      Path root,
      Path pointerPath,
      ChampionRole role,
      Path candidate,
      String evidenceRef,
      String parentCandidateId,
      String auditId,
      String auditProtocolId)
      throws IOException {
    ChampionPointer pointer =
        new ChampionPointer(
            POINTER_VERSION,
            role,
            EpsilonDecisionCheckpointManager.candidateId(candidate),
            root.relativize(candidate).toString(),
            EpsilonDecisionCheckpointManager.loadManifest(candidate).iteration,
            evidenceRef,
            parentCandidateId,
            auditId,
            auditProtocolId);
    writeJsonAtomic(pointerPath, GSON.toJson(pointer));
  }

  private static ChampionPointer loadChampionPointer(Path path) throws IOException {
    try {
      JsonObject json = JsonParser.parseString(Files.readString(path)).getAsJsonObject();
      requireFields(
          json,
          "pointerVersion",
          "role",
          "candidateId",
          "relativeCheckpoint",
          "iteration",
          "evidenceRef");
      ChampionPointer pointer = GSON.fromJson(json, ChampionPointer.class);
      if (pointer == null
          || pointer.pointerVersion() != POINTER_VERSION
          || pointer.role() == null
          || isBlank(pointer.candidateId())
          || isBlank(pointer.relativeCheckpoint())
          || pointer.iteration() < 0
          || isBlank(pointer.evidenceRef())
          || (pointer.role() == ChampionRole.ARENA
              && (pointer.parentCandidateId() != null
                  || pointer.auditId() != null
                  || pointer.auditProtocolId() != null))
          || !allNullOrNonBlank(
              pointer.parentCandidateId(), pointer.auditId(), pointer.auditProtocolId())) {
        throw new IOException("Invalid Decision champion pointer: " + path);
      }
      return pointer;
    } catch (IOException error) {
      throw error;
    } catch (RuntimeException error) {
      throw new IOException("Invalid Decision champion pointer: " + path, error);
    }
  }

  private static Path normalizeCandidate(Path root, Path candidate) throws IOException {
    Path normalized = normalizePath(candidate, "candidate");
    requireCandidateUnderRoot(root, normalized);
    EpsilonDecisionCheckpointManager.candidateId(normalized);
    return normalized;
  }

  private static <T> T withChampionLock(Path root, IoSupplier<T> operation) throws IOException {
    Path lockPath = root.resolve(".champion-pointer.lock");
    Files.createDirectories(root);
    try (FileChannel channel =
            FileChannel.open(lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        FileLock ignored = channel.lock()) {
      return operation.get();
    }
  }

  private static void publishProductionAudit(Path root, ProductionAudit audit) throws IOException {
    Path target = productionAuditPath(root, audit.auditId());
    if (Files.exists(target)) {
      requireSameProductionAudit(target, audit);
      return;
    }
    try {
      writeJsonCreateNewAtomic(target, GSON.toJson(audit));
    } catch (FileAlreadyExistsException error) {
      requireSameProductionAudit(target, audit);
    }
  }

  private static void requireProductionAudit(Path root, ChampionPointer pointer)
      throws IOException {
    ProductionAudit audit = loadProductionAudit(productionAuditPath(root, pointer.auditId()));
    if (!Objects.equals(audit.auditId(), pointer.auditId())
        || !Objects.equals(audit.protocolId(), pointer.auditProtocolId())
        || !Objects.equals(audit.candidateId(), pointer.candidateId())
        || !Objects.equals(audit.parentCandidateId(), pointer.parentCandidateId())
        || !Objects.equals(audit.evidenceRef(), pointer.evidenceRef())) {
      throw new IOException("Production champion pointer does not match its audit record");
    }
  }

  private static void requireSameProductionAudit(Path path, ProductionAudit expected)
      throws IOException {
    ProductionAudit actual = loadProductionAudit(path);
    if (!Objects.equals(actual.auditId(), expected.auditId())
        || !Objects.equals(actual.protocolId(), expected.protocolId())
        || !Objects.equals(actual.candidateId(), expected.candidateId())
        || !Objects.equals(actual.parentCandidateId(), expected.parentCandidateId())
        || !Objects.equals(actual.evidenceRef(), expected.evidenceRef())) {
      throw new IOException("Production audit ID is already bound to different evidence: " + path);
    }
  }

  private static ProductionAudit loadProductionAudit(Path path) throws IOException {
    if (!Files.isRegularFile(path)) {
      throw new IOException("Production audit record not found: " + path);
    }
    try {
      JsonObject json = JsonParser.parseString(Files.readString(path)).getAsJsonObject();
      requireFields(
          json,
          "auditVersion",
          "auditId",
          "protocolId",
          "candidateId",
          "parentCandidateId",
          "evidenceRef",
          "createdAt");
      ProductionAudit audit = GSON.fromJson(json, ProductionAudit.class);
      if (audit == null
          || audit.auditVersion() != PRODUCTION_AUDIT_VERSION
          || isBlank(audit.auditId())
          || isBlank(audit.protocolId())
          || isBlank(audit.candidateId())
          || isBlank(audit.parentCandidateId())
          || isBlank(audit.evidenceRef())
          || isBlank(audit.createdAt())) {
        throw new IOException("Invalid production audit record: " + path);
      }
      return audit;
    } catch (IOException error) {
      throw error;
    } catch (RuntimeException error) {
      throw new IOException("Invalid production audit record: " + path, error);
    }
  }

  static Path productionAuditPath(Path checkpointRoot, String auditId) throws IOException {
    Path root = normalizeRoot(checkpointRoot);
    validateAuditId(auditId);
    return root.resolve("production").resolve("audits").resolve(auditId + ".json");
  }

  private static void writeJsonAtomic(Path target, String json) throws IOException {
    Path parent = target.getParent();
    if (parent == null) {
      throw new IOException("Decision champion target has no parent: " + target);
    }
    Files.createDirectories(parent);
    Path temporary = Files.createTempFile(parent, "." + target.getFileName() + ".tmp-", ".json");
    boolean installed = false;
    try {
      Files.writeString(temporary, json + System.lineSeparator());
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        channel.force(true);
      }
      try {
        Files.move(
            temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException error) {
        throw new IOException("Atomic move is required: " + target, error);
      }
      installed = true;
    } finally {
      if (!installed) {
        Files.deleteIfExists(temporary);
      }
    }
  }

  private static void writeJsonCreateNewAtomic(Path target, String json) throws IOException {
    Path parent = target.getParent();
    if (parent == null) {
      throw new IOException("Decision champion target has no parent: " + target);
    }
    Files.createDirectories(parent);
    Path temporary = Files.createTempFile(parent, "." + target.getFileName() + ".tmp-", ".json");
    boolean installed = false;
    try {
      Files.writeString(temporary, json + System.lineSeparator());
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        channel.force(true);
      }
      Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
      installed = true;
    } catch (AtomicMoveNotSupportedException error) {
      throw new IOException("Atomic move is required: " + target, error);
    } finally {
      if (!installed) {
        Files.deleteIfExists(temporary);
      }
    }
  }

  private static Path resolveRelativeCandidate(Path root, String text, String label)
      throws IOException {
    Path relative;
    try {
      relative = Path.of(text);
    } catch (RuntimeException error) {
      throw new IOException("Invalid Decision " + label + " pointer path", error);
    }
    if (relative.isAbsolute()) {
      throw new IOException("Decision champion pointer path must be relative: " + relative);
    }
    Path candidate = root.resolve(relative).normalize();
    requireCandidateUnderRoot(root, candidate);
    return candidate;
  }

  private static void requireCandidateUnderRoot(Path root, Path candidate) throws IOException {
    Path candidateRoot = root.resolve("candidate").normalize();
    if (!candidate.startsWith(candidateRoot)
        || candidateRoot.relativize(candidate).getNameCount() != 2) {
      throw new IOException(
          "Decision candidate is not in the canonical checkpoint location: " + candidate);
    }
  }

  private static void validateAuditId(String auditId) throws IOException {
    if (auditId == null || !AUDIT_ID.matcher(auditId).matches()) {
      throw new IOException("Invalid Decision production auditId: " + auditId);
    }
  }

  private static void requireFields(JsonObject json, String... fields) {
    for (String field : fields) {
      if (!json.has(field) || json.get(field).isJsonNull()) {
        throw new IllegalArgumentException("Decision champion field is missing: " + field);
      }
    }
  }

  private static void requireNonBlank(String value, String name) throws IOException {
    if (isBlank(value)) {
      throw new IOException("Decision champion " + name + " must not be blank");
    }
  }

  private static boolean isBlank(String value) {
    return value == null || value.isBlank();
  }

  private static boolean allNullOrNonBlank(String... values) {
    boolean allNull = true;
    for (String value : values) {
      allNull &= value == null;
    }
    if (allNull) {
      return true;
    }
    for (String value : values) {
      if (isBlank(value)) {
        return false;
      }
    }
    return true;
  }

  private static Path normalizeRoot(Path root) throws IOException {
    return normalizePath(root, "checkpoint root");
  }

  private static Path normalizePath(Path path, String label) throws IOException {
    if (path == null) {
      throw new IOException("Decision " + label + " path is null");
    }
    return path.toAbsolutePath().normalize();
  }

  private enum ChampionRole {
    ARENA,
    PRODUCTION
  }

  private record ChampionPointer(
      int pointerVersion,
      ChampionRole role,
      String candidateId,
      String relativeCheckpoint,
      int iteration,
      String evidenceRef,
      String parentCandidateId,
      String auditId,
      String auditProtocolId) {}

  private record ProductionAudit(
      int auditVersion,
      String auditId,
      String protocolId,
      String candidateId,
      String parentCandidateId,
      String evidenceRef,
      String createdAt) {}

  @FunctionalInterface
  private interface IoSupplier<T> {
    T get() throws IOException;
  }
}
