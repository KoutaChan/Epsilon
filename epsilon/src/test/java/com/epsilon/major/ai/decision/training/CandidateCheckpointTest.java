package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 候補の保存と採用参照を、チェックポイント本体だけで復元・検証する。 */
public class CandidateCheckpointTest {
  private static final Gson JSON = new Gson();

  @Test
  public void explicitCheckpointResolvesWithoutPromotionAndRejectsMissingModel() throws Exception {
    Path root = Files.createTempDirectory("decision-explicit-");
    try {
      Path checkpoint = writeCheckpoint(root.resolve("pretrained"), 3);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpoint), checkpoint);
      Files.delete(checkpoint.resolve(EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FILE));
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpoint));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void savedCandidateResumesIdenticalSourceAndRejectsChangedWeights() throws Exception {
    Path root = Files.createTempDirectory("decision-candidate-");
    try {
      Path source = writeCheckpoint(root.resolve("source"), 4);
      Path candidate = EpsilonDecisionCheckpointManager.saveCandidate(root, "run-a", source);
      Assert.assertEquals(candidate, root.resolve("candidate/run-a/00004"));
      try (var files = Files.list(candidate)) {
        Assert.assertEquals(
            files.map(path -> path.getFileName().toString()).collect(Collectors.toSet()),
            Set.of(
                "manifest.json",
                "architecture.id",
                EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FILE));
      }
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.saveCandidate(root, "run-a", source), candidate);
      Path candidateModel = candidate.resolve(EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FILE);
      byte[] originalWeights = Files.readAllBytes(candidateModel);
      Files.write(
          source.resolve(EpsilonDecisionCheckpointBundle.CURRENT_MODEL_FILE), new byte[] {7, 8, 9});
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.saveCandidate(root, "run-a", source));
      Assert.assertEquals(Files.readAllBytes(candidateModel), originalWeights);
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void versionThreePointersResolveFromCheckpointWithoutAdditionalMetadata()
      throws Exception {
    Path root = Files.createTempDirectory("decision-pointer-");
    try {
      Path champion =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source"), 2));
      writePointer(root, "ARENA", champion, "initial/00002", 2);
      writePointer(root, "PRODUCTION", champion, "initial/00002", 2);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root), champion);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root), champion);
      Assert.assertEquals(EpsilonDecisionCheckpointManager.resolveExistingStrict(root), champion);
    } finally {
      deleteTree(root);
    }
  }

  @DataProvider
  public Object[][] invalidPointerFields() {
    return new Object[][] {
      {"candidateId", "another/00002"},
      {"iteration", 3},
      {"relativeCheckpoint", "../outside/00002"},
      {"role", "PRODUCTION"}
    };
  }

  @Test(dataProvider = "invalidPointerFields")
  public void invalidPointerIdentityIterationOrLocationIsRejected(String field, Object value)
      throws Exception {
    Path root = Files.createTempDirectory("decision-invalid-pointer-");
    try {
      Path champion =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source"), 2));
      JsonObject pointer = writePointer(root, "ARENA", champion, "initial/00002", 2);
      if (value instanceof Number number) {
        pointer.addProperty(field, number);
      } else {
        pointer.addProperty(field, value.toString());
      }
      Files.writeString(root.resolve("arena/current.json"), JSON.toJson(pointer));
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void arenaPromotionKeepsProductionAndRejectsSkippedIteration() throws Exception {
    Path root = Files.createTempDirectory("decision-arena-promotion-");
    try {
      Path initial =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source-2"), 2));
      Path next =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "next", writeCheckpoint(root.resolve("source-3"), 3));
      Path skipped =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "skipped", writeCheckpoint(root.resolve("source-5"), 5));
      writePointer(root, "ARENA", initial, "initial/00002", 2);
      writePointer(root, "PRODUCTION", initial, "initial/00002", 2);
      byte[] productionPointer = Files.readAllBytes(root.resolve("production/current.json"));

      EpsilonDecisionChampionStore.promoteArenaChampion(root, next, "duel passed");
      EpsilonDecisionChampionStore.promoteArenaChampion(root, next, "duel passed");
      Assert.assertEquals(EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root), next);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root), initial);
      Assert.assertEquals(
          Files.readAllBytes(root.resolve("production/current.json")), productionPointer);
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionChampionStore.promoteArenaChampion(root, skipped, "duel passed"));
      Assert.assertEquals(EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root), next);
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void productionPromotionRequiresCurrentParentAndPersistsIndependentAudit()
      throws Exception {
    Path root = Files.createTempDirectory("decision-production-promotion-");
    try {
      Path initial =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source-2"), 2));
      Path next =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "next", writeCheckpoint(root.resolve("source-3"), 3));
      writePointer(root, "ARENA", initial, "initial/00002", 2);
      writePointer(root, "PRODUCTION", initial, "initial/00002", 2);
      EpsilonDecisionChampionStore.promoteArenaChampion(root, next, "duel passed");
      Assert.expectThrows(
          IOException.class,
          () ->
              EpsilonDecisionChampionStore.promoteProductionChampion(
                  root, next, "fixed wall passed", "wrong/00002", "audit-1", "protocol-1"));
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root), initial);
      EpsilonDecisionChampionStore.promoteProductionChampion(
          root, next, "fixed wall passed", "initial/00002", "audit-1", "protocol-1");
      EpsilonDecisionChampionStore.promoteProductionChampion(
          root, next, "fixed wall passed", "initial/00002", "audit-1", "protocol-1");
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root), next);
      Assert.assertEquals(EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root), next);
      Files.delete(root.resolve("production/audits/audit-1.json"));
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void snapshotRegistryRestoresCheckpointIdentityWithoutAdditionalMetadata()
      throws Exception {
    Path root = Files.createTempDirectory("decision-candidate-snapshot-");
    try {
      Path champion =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source"), 2));
      EpsilonDecisionSnapshotPool pool = new EpsilonDecisionSnapshotPool(2);
      pool.registerArenaChampion(champion);
      pool.save(root);
      EpsilonDecisionSnapshotPool restored = EpsilonDecisionSnapshotPool.load(root, 2);
      long id = restored.sampleOpponentIdsForMacro(17, 0.7)[0][0];
      Assert.assertEquals(restored.snapshot(id).path(), champion.toString());
      Assert.assertEquals(restored.snapshot(id).iteration(), 2);
      Assert.assertTrue(restored.snapshot(id).currentArenaChampion());
      Files.writeString(champion.resolve("architecture.id"), "incompatible-architecture");
      Assert.expectThrows(IOException.class, () -> EpsilonDecisionSnapshotPool.load(root, 2));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void parentTraversalRunIdIsRejectedWithoutWritingCheckpoint() throws Exception {
    Path root = Files.createTempDirectory("decision-invalid-run-");
    try {
      Path source = writeCheckpoint(root.resolve("source"), 2);
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.saveCandidate(root, "..", source));
      Assert.assertFalse(Files.exists(root.resolve("00002")));
      Assert.assertFalse(Files.exists(root.resolve("candidate")));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void negativeIterationIsRejectedWithoutCreatingCandidate() throws Exception {
    Path root = Files.createTempDirectory("decision-negative-iteration-");
    try {
      Path source = writeCheckpoint(root.resolve("source"), -1);
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.saveCandidate(root, "run", source));
      Assert.assertFalse(Files.exists(root.resolve("candidate")));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void pointerCannotUseValidCandidateNestedUnderAnotherRoot() throws Exception {
    Path root = Files.createTempDirectory("decision-nested-candidate-");
    try {
      Path nested = writeCheckpoint(root.resolve("candidate/nested/candidate/run/00002"), 2);
      EpsilonDecisionCheckpointManager.requireValidCheckpoint(nested);
      writePointer(root, "ARENA", nested, "run/00002", 2);
      Assert.expectThrows(
          IOException.class,
          () -> EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root));
    } finally {
      deleteTree(root);
    }
  }

  @Test
  public void guardFailureJournalIsReusedWithoutEvaluatingOrChangingChampions() throws Exception {
    Path root = Files.createTempDirectory("decision-duel-resume-");
    try {
      Path champion =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "initial", writeCheckpoint(root.resolve("source-2"), 2));
      Path candidate =
          EpsilonDecisionCheckpointManager.saveCandidate(
              root, "next", writeCheckpoint(root.resolve("source-3"), 3));
      writePointer(root, "ARENA", champion, "initial/00002", 2);
      writePointer(root, "PRODUCTION", champion, "initial/00002", 2);
      Path report = root.resolve("report");
      DecisionSelectedPgDuelRunner.Resolution rejected =
          DecisionSelectedPgDuelRunner.resolve(
              duelRequest(root, report, candidate, champion, Optional.of("macro KL exceeded")),
              null,
              null);
      Path journal = report.resolve("duel-resolution.json");
      byte[] originalJournal = Files.readAllBytes(journal);
      DecisionSelectedPgDuelRunner.Resolution resumed =
          DecisionSelectedPgDuelRunner.resolve(
              duelRequest(root, report, candidate, champion, Optional.empty()), null, null);

      Assert.assertEquals(rejected.status(), DecisionSelectedPgDuelResult.Status.GUARD_FAILED);
      Assert.assertEquals(resumed, rejected);
      Assert.assertEquals(Files.readAllBytes(journal), originalJournal);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(root), champion);
      Assert.assertEquals(
          EpsilonDecisionCheckpointManager.resolveProductionChampionStrict(root), champion);
      Assert.expectThrows(
          IOException.class,
          () ->
              DecisionSelectedPgDuelRunner.resolve(
                  duelRequest(root, report, champion, champion, Optional.empty()), null, null));
    } finally {
      deleteTree(root);
    }
  }

  private static DecisionSelectedPgDuelRunner.Request duelRequest(
      Path root, Path report, Path candidate, Path champion, Optional<String> guardFailure) {
    return new DecisionSelectedPgDuelRunner.Request(
        root,
        report,
        candidate,
        champion,
        1,
        3,
        1,
        123L,
        1,
        2,
        guardFailure,
        new DecisionChampionDuelSettings(1, 2, 0.01, 0.25));
  }

  private static Path writeCheckpoint(Path directory, int iteration) throws IOException {
    Files.createDirectories(directory);
    EpsilonDecisionCheckpointBundle manifest =
        new EpsilonDecisionCheckpointBundle(11, iteration, 29, 16, EpsilonUtilityProfile.TENHOU);
    Files.writeString(directory.resolve("manifest.json"), JSON.toJson(manifest));
    Files.writeString(directory.resolve("architecture.id"), manifest.architecture);
    Files.write(directory.resolve(manifest.modelFile), new byte[] {1, 2, 3});
    return directory;
  }

  private static JsonObject writePointer(
      Path root, String role, Path checkpoint, String candidateId, int iteration)
      throws IOException {
    JsonObject pointer = new JsonObject();
    pointer.addProperty("pointerVersion", 3);
    pointer.addProperty("role", role);
    pointer.addProperty("candidateId", candidateId);
    pointer.addProperty("relativeCheckpoint", root.relativize(checkpoint).toString());
    pointer.addProperty("iteration", iteration);
    pointer.addProperty("evidenceRef", "initial checkpoint");
    Path directory = root.resolve(role.equals("ARENA") ? "arena" : "production");
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("current.json"), JSON.toJson(pointer));
    return pointer;
  }

  private static void deleteTree(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(path);
      }
    }
  }
}
