package com.epsilon.prism.play;

import com.epsilon.agent.observe.Observation;
import java.io.IOException;
import java.util.HexFormat;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 配布モデルの参照スコアのファイルを、実行のバックエンドごとに選べることと、ファイルの行の数を確かめる。 */
public class PublishedScoreReferenceTest {
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
