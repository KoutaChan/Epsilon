package com.epsilon.prism.play;

import com.epsilon.agent.observe.Observation;
import java.io.IOException;
import java.util.HexFormat;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 配布モデルの参照スコアのファイルを、実行のバックエンドごとに選べることと、ファイルの行の数を確かめる。 */
public class PublishedScoreReferenceTest {
  @Test
  public void eachBackendSelectsItsPublishedScores() {
    Assert.assertEquals(
        PublishedScoreReference.resource("CUDA"), "published-cuda-scores.properties");
    Assert.assertEquals(PublishedScoreReference.resource("EAGER"), "published-scores.properties");
    Assert.assertEquals(PublishedScoreReference.resource("HIP"), "published-scores.properties");
    Assert.expectThrows(
        IllegalArgumentException.class, () -> PublishedScoreReference.resource("NATIVE"));
  }

  @Test
  public void originalReferenceContainsBothRulesAndAllCpuAndHipRows() throws IOException {
    var reference = PublishedScoreReference.load("HIP");
    for (Object[] models :
        new Object[][] {PublishedModel.fourPlayerModels(), PublishedModel.threePlayerModels()}) {
      for (Object item : models) {
        var model = (PublishedModel) item;
        Assert.assertEquals(
            PublishedPositions.fingerprint(PublishedPositions.sample(model.variant())),
            reference.getProperty(model.variant() + ".inputs"));
        for (String backend : new String[] {"EAGER.FLOAT32", "EAGER.BFLOAT16", "HIP.BFLOAT16"}) {
          Assert.assertEquals(
              reference.getProperty(model.directory() + "." + backend).split(";", -1).length, 32);
        }
        Assert.assertNull(reference.getProperty(model.directory() + ".CUDA.BFLOAT16"));
      }
    }
  }

  @Test
  public void cudaReferenceMatchesEveryPublishedLegalActionCount() throws IOException {
    var reference = PublishedScoreReference.load("CUDA");
    for (Object[] models :
        new Object[][] {PublishedModel.fourPlayerModels(), PublishedModel.threePlayerModels()}) {
      for (Object item : models) {
        var model = (PublishedModel) item;
        var rows = PublishedPositions.sample(model.variant());
        var row = new Observation(model.variant());
        Assert.assertEquals(
            reference.getProperty(model.variant() + ".inputs"),
            PublishedPositions.fingerprint(rows));
        Assert.assertNull(reference.getProperty(model.directory() + ".HIP.BFLOAT16"));
        var scores = reference.getProperty(model.directory() + ".CUDA.BFLOAT16").split(";", -1);
        Assert.assertEquals(scores.length, rows.size());
        for (int i = 0; i < rows.size(); i++) {
          var bits = scores[i].trim().split("\\s+");
          Assert.assertEquals(bits.length, row.bind(rows, i).legalActions());
          for (String word : bits) {
            Assert.assertEquals(word.length(), 8);
            Assert.assertTrue(Float.isFinite(Float.intBitsToFloat(HexFormat.fromHexDigits(word))));
          }
        }
      }
    }
  }
}
