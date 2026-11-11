package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.SettingsException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.OptionalLong;
import org.testng.annotations.Test;

/** 要求のファイルの番号の付け方と読み方、プロセスの起動時刻の読み方、jsonl への追記を確かめる。 */
public class ControlFilesTest {

  @Test
  public void requestNumbersCountOnAcrossRetiredRequests() throws Exception {
    RunDirectory run = new RunDirectory(Files.createTempDirectory("control"));
    ControlRequests requests = new ControlRequests(run);
    ControlRequest stop = requests.submit(new Request.Stop(Request.At.JOBS));
    ControlRequest pause = requests.submit(new Request.Pause("COLLECT"));
    assertEquals(stop.fileName(), "1-stop.json");
    assertEquals(pause.fileName(), "2-pause.json");
    requests.retire(stop);
    requests.retire(pause);
    ControlRequest restart = requests.submit(new Request.Restart(Request.At.STEP));
    assertEquals(restart.number(), 3);
    assertEquals(requests.pending(), List.of(restart));
    assertEquals(RetiredRequests.read(run), List.of(stop, pause));
    assertFalse(requests.stopped());
    assertEquals(requests.stoppedReason(), "");
    // 要求のファイルは JSON で、要求の種類は kind に書く
    String text = Files.readString(run.control().resolve("3-restart.json"), StandardCharsets.UTF_8);
    Node.Table tree = Json.read(text, "request");
    assertEquals(((Node.Table) tree.get("request")).get("kind"), new Node.Text("RESTART", "expected"));
  }

  @Test
  public void aGrpRefitRequestRoundTripsAndRetiresOnce() throws Exception {
    RunDirectory run = new RunDirectory(Files.createTempDirectory("control-grp-refit"));
    ControlRequests requests = new ControlRequests(run);
    ControlRequest refit = requests.submit(new Request.GrpRefit(5));
    assertEquals(refit.fileName(), "1-grp-refit.json");
    assertEquals(requests.pending(), List.of(refit));
    String text = Files.readString(run.control().resolve("1-grp-refit.json"), StandardCharsets.UTF_8);
    Node.Table tree = (Node.Table) Json.read(text, "request").get("request");
    assertEquals(tree.get("kind"), new Node.Text("GRP_REFIT", "expected"));
    // 安全点の確かめと GRP の裏の作業が同じ頼みを移しても誤りにならない
    requests.retire(refit);
    requests.retire(refit);
    assertEquals(requests.pending(), List.of());
    assertEquals(RetiredRequests.read(run), List.of(refit));
    assertEquals(requests.submit(new Request.Resume()).number(), 2);
  }

  @Test
  public void aRequestFileWithAnotherRequestInsideIsAnError() throws Exception {
    RunDirectory run = new RunDirectory(Files.createTempDirectory("control-mismatch"));
    ControlRequests requests = new ControlRequests(run);
    requests.submit(new Request.Resume());
    Files.move(run.control().resolve("1-resume.json"), run.control().resolve("4-resume.json"));
    expectThrows(SettingsException.class, requests::pending);
    expectThrows(IllegalArgumentException.class, () -> new Request.Restart(Request.At.JOBS));
  }

  @Test
  public void theStartTimeIsTheTwentySecondFieldAfterTheName() {
    String fields = " S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 424242 20 21";
    assertEquals(ProcessStart.parse("123 (java) " + fields), OptionalLong.of(424242));
    assertEquals(ProcessStart.parse("123 (a b) c)) " + fields), OptionalLong.of(424242));
    assertEquals(ProcessStart.parse("123 (gone) Z" + fields.substring(2)), OptionalLong.empty());
    assertEquals(ProcessStart.parse("123 (short) S 1 2"), OptionalLong.empty());
    assertEquals(ProcessStart.parse("no name"), OptionalLong.empty());
    long pid = ProcessHandle.current().pid();
    OptionalLong start = ProcessStart.of(pid);
    assertTrue(start.isPresent());
    assertTrue(ProcessStart.alive(pid, start.getAsLong()));
    assertFalse(ProcessStart.alive(pid, start.getAsLong() + 1));
  }

  @Test
  public void aHalfWrittenLastLineIsDroppedAndLinesAreNotDoubled() throws Exception {
    Path file = Files.createTempDirectory("jsonl").resolve("reports.jsonl");
    Node.Table first = Json.read("{\"iteration\":1,\"x\":1}", "line");
    Node one = first.get("iteration");
    assertTrue(DurableFiles.appendAfterLast(file, "iteration", one, first));
    assertFalse(DurableFiles.appendAfterLast(file, "iteration", one, first));
    Files.writeString(file, "{\"iteration\":2,\"x\"", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    assertEquals(DurableFiles.lines(file), List.of("{\"iteration\":1,\"x\":1}"));
    Node.Table second = Json.read("{\"iteration\":2,\"x\":2}", "line");
    assertTrue(DurableFiles.appendAfterLast(file, "iteration", second.get("iteration"), second));
    assertEquals(
        Files.readString(file, StandardCharsets.UTF_8), "{\"iteration\":1,\"x\":1}\n{\"iteration\":2,\"x\":2}\n");
    assertFalse(DurableFiles.appendOnce(file, "iteration", one, first));
    assertEquals(DurableFiles.lines(file).size(), 2);
  }

  @Test
  public void aSealedOutputIsNeverSealedTwice() throws Exception {
    Path directory = Files.createTempDirectory("seal");
    Path part = directory.resolve("COLLECT.part");
    DurableFiles.write(part.resolve("report.json"), "{}\n");
    DurableFiles.seal(part, directory.resolve("COLLECT"));
    assertTrue(Files.exists(directory.resolve("COLLECT").resolve("report.json")));
    DurableFiles.write(part.resolve("report.json"), "{}\n");
    expectThrows(FileAlreadyExistsException.class, () -> DurableFiles.seal(part, directory.resolve("COLLECT")));
  }
}
