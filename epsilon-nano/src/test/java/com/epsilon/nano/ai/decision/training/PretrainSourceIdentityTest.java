package com.epsilon.nano.ai.decision.training;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜キャッシュは内容検証ではなくファイルの属性で識別する。 */
public class PretrainSourceIdentityTest {
  @Test
  public void sourceOrderDoesNotChangeIdentity() throws Exception {
    Path first = Files.createTempFile("pretrain-source-", ".json");
    Path second = Files.createTempFile("pretrain-source-", ".json");
    try {
      Assert.assertEquals(
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(first, second)),
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(second, first)));
    } finally {
      Files.delete(first);
      Files.delete(second);
    }
  }

  @Test
  public void sizeAndModificationTimeChangeIdentityWithoutHashingContents() throws Exception {
    Path source = Files.createTempFile("pretrain-source-", ".json");
    try {
      Files.writeString(source, "first");
      FileTime modified = Files.getLastModifiedTime(source);
      String original = EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(source));

      Files.writeString(source, "other");
      Files.setLastModifiedTime(source, modified);
      Assert.assertEquals(
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(source)), original);

      Files.setLastModifiedTime(source, FileTime.fromMillis(modified.toMillis() + 10_000));
      Assert.assertNotEquals(
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(source)), original);

      Files.writeString(source, "longer content");
      Files.setLastModifiedTime(source, modified);
      Assert.assertNotEquals(
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(source)), original);
    } finally {
      Files.delete(source);
    }
  }

  @Test
  public void differentPathsHaveDifferentIdentities() throws Exception {
    Path first = Files.createTempFile("pretrain-source-", ".json");
    Path second = Files.createTempFile("pretrain-source-", ".json");
    try {
      Files.setLastModifiedTime(second, Files.getLastModifiedTime(first));
      Assert.assertNotEquals(
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(first)),
          EpsilonDecisionPretrainDatasetCompiler.sourceDigest(List.of(second)));
    } finally {
      Files.delete(first);
      Files.delete(second);
    }
  }
}
