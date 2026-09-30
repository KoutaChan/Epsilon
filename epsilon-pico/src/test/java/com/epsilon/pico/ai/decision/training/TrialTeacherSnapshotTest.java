package com.epsilon.pico.ai.decision.training;

import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 固定済みの教師モデルを内容の再走査なしで引き継ぐ。 */
public class TrialTeacherSnapshotTest {
  @Test
  public void disabledTeacherDoesNotAccessFiles() throws Exception {
    Assert.assertFalse(DecisionKlTargetTrial.snapshotTeacher(null, null, false).enabled());
  }

  @Test
  public void existingTeacherIsReusedWithoutHashVerification() throws Exception {
    Path trial = Files.createTempDirectory("trial-teacher-");
    Path teacher = Files.createDirectory(trial.resolve("grp"));
    Path parameters = teacher.resolve("parameters");
    try {
      Files.writeString(parameters, "existing snapshot");
      Assert.assertTrue(DecisionKlTargetTrial.snapshotTeacher(null, trial, true).enabled());
      Assert.assertEquals(Files.readString(parameters), "existing snapshot");
    } finally {
      Files.deleteIfExists(parameters);
      Files.delete(teacher);
      Files.delete(trial);
    }
  }
}
