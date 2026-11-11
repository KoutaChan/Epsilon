package com.epsilon.agent.app;

import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run の control に要求を 1 つ置くコマンド。段とジョブを動かす JVM は、その要求が効く次のセーフポイントで従う。
 * JVM が動いていなければ、次に起動したときの最初のセーフポイントで従う。
 */
public sealed interface ControlCommand extends BuiltinCommand {

  /** run のディレクトリ。 */
  Path run();

  /** 置く要求。 */
  Request toRequest();

  @Override
  default Outcome run(Context<Record> context) throws Exception {
    context.requireNoLayers();
    return Submission.submit(new RunDirectory(run()), toRequest());
  }

  /** 要求を置く手続き。 */
  final class Submission {

    private static final Logger log = LoggerFactory.getLogger(ControlCommand.class);

    private Submission() {}

    /**
     * 要求を置く。
     *
     * @throws SettingsException run に設定の行が無い。または、取り消す要求が、まだ従っていない要求の中に無い
     */
    static Outcome submit(RunDirectory directory, Request request) throws IOException {
      if (!Files.isRegularFile(directory.settings())) {
        throw new SettingsException(
            List.of("The run " + directory.root() + " has no settings rows: create it with settings create"));
      }
      ControlRequests requests = new ControlRequests(directory);
      if (request instanceof Request.Cancel cancel
          && requests.pending().stream().noneMatch(pending -> pending.number() == cancel.request())) {
        throw new SettingsException(List.of("No pending request " + cancel.request() + " in " + directory.control()));
      }
      ControlRequest written = requests.submit(request);
      log.info("Request {} ({}) waits in {}", written.number(), Status.describe(written.request()), directory.control());
      return Outcome.ok("request " + written.number() + ": " + Status.describe(written.request()));
    }
  }

  /**
   * 止める要求。もう一度起動すれば、止めた所から続ける。
   *
   * @param at 止める時点
   */
  @Named(value = "control stop", doc = "Request a stop at the next safe point, after the iteration closes, or after its jobs finish")
  record Stop(
      @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
      @Key(doc = "When to stop (STEP: the next safe point, ITERATION: after the iteration closes, JOBS: after all its jobs finish)",
              presence = Presence.REQUIRED)
          Request.At at)
      implements ControlCommand {

    @Override
    public Request toRequest() {
      return new Request.Stop(at);
    }
  }

  /**
   * 起動し直す要求。速さに関わるキーの変更を効かせるのに使う。
   *
   * @param at 起動し直す時点
   */
  @Named(value = "control restart", doc = "Request a restart at the next safe point or after the iteration closes")
  record Restart(
      @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
      @Key(doc = "When to restart (STEP: the next safe point, ITERATION: after the iteration closes)", presence = Presence.REQUIRED) Request.At at)
      implements ControlCommand {

    public Restart {
      if (at == Request.At.JOBS) {
        throw new InvalidSetting("at", "a restart takes effect at STEP or ITERATION, not JOBS");
      }
    }

    @Override
    public Request toRequest() {
      return new Request.Restart(at);
    }
  }

  /**
   * 一時停止の要求。再開の要求で消すまで、セーフポイントで待つ。
   *
   * @param before この段の前でだけ待つ。指定しなければ次のセーフポイントで待つ
   */
  @Named(value = "control pause", doc = "Request a pause at the next safe point or before a step (cleared by control resume)")
  record Pause(
      @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
      @Key(doc = "Wait only before this step (without it, wait at the next safe point)", presence = Presence.ABSENT) String before)
      implements ControlCommand {

    @Override
    public Request toRequest() {
      return new Request.Pause(before);
    }
  }

  /** 再開の要求。それより前に置いた一時停止の要求を消す。 */
  @Named(value = "control resume", doc = "Request that the earlier pause requests be cleared")
  record Resume(@Key(doc = "Run directory", presence = Presence.REQUIRED) Path run) implements ControlCommand {

    @Override
    public Request toRequest() {
      return new Request.Resume();
    }
  }

  /**
   * まだ従っていない要求を 1 つ消す要求。
   *
   * @param request 消す要求の番号
   */
  @Named(value = "control cancel", doc = "Request that one pending request be cancelled by number")
  record Cancel(
      @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
      @Key(doc = "Number of the request to cancel", presence = Presence.REQUIRED) @Range(min = 1) int request)
      implements ControlCommand {

    @Override
    public Request toRequest() {
      return new Request.Cancel(request);
    }
  }
}
