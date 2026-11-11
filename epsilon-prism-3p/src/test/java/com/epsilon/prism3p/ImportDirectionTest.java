package com.epsilon.prism3p;

import com.epsilon.prism.ImportDirection;
import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** prism-3p の依存の向きのテスト。main とテストのコードは classic と prism-4p を import せず、ワイルドカードの import も使わない。pom では classic を runtime の scope でだけ使う。 */
public class ImportDirectionTest {

  @Test
  public void importsNoClassicNorTheFourPlayerModule() throws IOException {
    Assert.assertEquals(
        ImportDirection.violations(ImportDirection.CLASSIC_PACKAGES + "|com\\.epsilon\\.prism4p(\\..*)?"), List.of());
  }

  @Test
  public void pomNamesClassicOnlyAtRuntime() throws IOException {
    Assert.assertEquals(ImportDirection.pomViolations(), List.of());
  }

  @Test
  public void classpathHoldsNoFourPlayerModule() {
    Assert.assertEquals(ImportDirection.loadable(List.of("com.epsilon.prism4p.variant.FourPlayer")), List.of());
  }
}
