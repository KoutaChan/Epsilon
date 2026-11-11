package com.epsilon.agent.app;

import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunStatus;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run の状態（status.json）を、ハートビートの古さ、pid の生死、まだ従っていない要求と一緒に出す。動いているはずの JVM の
 * ハートビートが古いか、pid のプロセスがいなければ、きれいに止まっていないと出す。何も止めはしない。
 *
 * @param run run のディレクトリ
 */
@Named(value = "status", doc = "Print the run state, the heartbeat age, whether the pid is alive, and the pending requests")
public record Status(@Key(doc = "Run directory", presence = Presence.REQUIRED) Path run) implements BuiltinCommand {

  private static final Logger log = LoggerFactory.getLogger(Status.class);

  @Override
  public Outcome run(Context<Record> context) throws Exception {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    if (!Files.isRegularFile(directory.settings())) {
      throw new SettingsException(List.of("The run " + run + " has no settings rows: create it with settings create"));
    }
    ControlRequests requests = new ControlRequests(directory);
    List<ControlRequest> pending = requests.pending();
    PrintStream out = context.out();
    if (!Files.exists(directory.status())) {
      out.println("No JVM has run " + directory.root() + " yet");
      printRequests(out, pending, requests);
      return Outcome.ok("never started");
    }
    RunStatus status = RunStatus.read(directory.status());
    Instant now = Instant.now();
    out.println(
        "state " + status.state() + " since " + status.since() + " (pid " + status.pid() + ", jar " + status.jar() + ")");
    if (status.iteration() != null) {
      StringBuilder line = new StringBuilder("iteration " + status.iteration());
      if (status.step() != null) {
        line.append(", step ").append(status.step());
      }
      if (status.job() != null) {
        line.append(", job ").append(status.job());
      }
      if (status.settingsRow() != null) {
        line.append(", settings row ").append(status.settingsRow());
      }
      if (status.launchRow() != null) {
        line.append(" (launched with row ").append(status.launchRow()).append(')');
      }
      out.println(line);
    }
    out.println("updated " + status.updated() + " (" + Duration.between(status.updated(), now).toSeconds() + " s ago)");
    if (!status.pendingRestart().isEmpty()) {
      out.println("takes effect after a restart: " + String.join(", ", status.pendingRestart()));
    }
    if (status.reason() != null) {
      out.println("reason: " + status.reason());
    }
    printRequests(out, pending, requests);
    if (status.unclean(now)) {
      String warning =
          "The JVM of " + directory.root() + " did not stop cleanly: it was " + status.state() + " at "
              + status.updated() + " and its heartbeat or process is gone";
      out.println(warning);
      log.warn("{}", warning);
      return Outcome.ok(status.state() + ", not stopped cleanly");
    }
    return Outcome.ok(status.state().toString());
  }

  private static void printRequests(PrintStream out, List<ControlRequest> pending, ControlRequests requests)
      throws Exception {
    for (ControlRequest request : pending) {
      out.println("pending request " + request.number() + ": " + describe(request.request()) + " (" + request.time() + ")");
    }
    if (requests.stopped()) {
      out.println("stopped mark: " + requests.stoppedReason());
    }
  }

  /** 要求を 1 行で書く。 */
  public static String describe(Request request) {
    return switch (request) {
      case Request.Stop stop -> "stop at " + stop.at();
      case Request.Restart restart -> "restart at " + restart.at();
      case Request.Pause pause -> pause.before() == null ? "pause" : "pause before " + pause.before();
      case Request.Resume resume -> "resume";
      case Request.Cancel cancel -> "cancel request " + cancel.request();
      case Request.GrpRefit refit -> "GRP refit at slot " + refit.slot();
    };
  }
}
