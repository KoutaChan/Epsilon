package com.epsilon.prism4p;

import com.epsilon.prism.ImportDirection;
import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * prism-4p の依存の向きのテスト。main とテストのコードは classic を import せず、ワイルドカードの import も使わない。pom に書く classic は
 * runtime の scope だけ。
 */
public class ImportDirectionTest {

  @Test
  public void importsNoClassic() throws IOException {
    Assert.assertEquals(ImportDirection.violations(ImportDirection.CLASSIC_PACKAGES), List.of());
  }

  @Test
  public void pomNamesClassicOnlyAtRuntime() throws IOException {
    Assert.assertEquals(ImportDirection.pomViolations(), List.of());
  }
}
