package com.epsilon.prism.run;

import com.epsilon.prism.run.StopSignal.SafePoint;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 止める合図: SIGTERM（2 回でも）は合図を立てるだけで、子の JVM は自分の安全点まで進んで 0 で終わる（主クラスは {@link StopSignalProbe}）。
 * 試験の合図は指定の安全点の番号で立ち、立った合図はどの安全点でも止める。
 */
public class StopSignalTest {

  @Test(timeOut = 120_000)
  public void termOnlyRaisesTheFlag() throws Exception {
    if (System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win"))
      throw new SkipException("SIGTERM is a POSIX signal");
    Path java = Path.of(System.getProperty("java.home"), "bin", "java");
    List<String> command = new ArrayList<>(List.of(java.toString(), "-Xmx64m", "-XX:TieredStopAtLevel=1"));
    command.addAll(List.of("-cp", System.getProperty("java.class.path"), StopSignalProbe.class.getName()));
    Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
    try (var output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
      String line;
      while ((line = output.readLine()) != null && !line.equals("ready")) {
        // 起動の記録は読み飛ばす
      }
      Assert.assertEquals(line, "ready");
      // ProcessHandle.destroy は SIGTERM を送るだけ（Process.destroy と違って出力の流れを閉じない）
      process.toHandle().destroy();
      Thread.sleep(100);
      process.toHandle().destroy();
      List<String> rest = new ArrayList<>();
      while ((line = output.readLine()) != null) rest.add(line);
      Assert.assertTrue(process.waitFor(60, TimeUnit.SECONDS));
      Assert.assertEquals(process.exitValue(), 0, String.join("\n", rest));
      Assert.assertTrue(rest.contains("stopped"), String.join("\n", rest));
    } finally {
      process.destroyForcibly();
    }
  }

  @Test
  public void testSignalsStopAtTheirSafePoint() {
    StopSignal stop = StopSignal.at(SafePoint.ITERATED, 3);
    Assert.assertFalse(stop.stopsAt(SafePoint.SEALED, 3));
    Assert.assertFalse(stop.stopsAt(SafePoint.ITERATED, 2));
    Assert.assertFalse(stop.requested());
    Assert.assertTrue(stop.stopsAt(SafePoint.ITERATED, 3));
    Assert.assertTrue(stop.stopsAt(SafePoint.MEASURED, 8));
    StopSignal none = StopSignal.none();
    Assert.assertFalse(none.stopsAt(SafePoint.SEALED, 1));
    none.request("test");
    none.request("test");
    Assert.assertTrue(none.requested() && none.stopsAt(SafePoint.SEALED, 1));
  }
}
