package com.epsilon.agent.app;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import java.nio.file.Path;
import java.util.List;

/** 入口のテストに使う小さなアプリ。名前は demo で、プリセットはテストのクラスパスの demo/presets に置く。 */
final class DemoApp {

  private DemoApp() {}

  /** demo の全体の設定。 */
  record Demo(
      @Key(doc = "同時に進める半荘の数", value = "8", change = Change.MACHINE) @Range(min = 1) int workers,
      @Key(doc = "乱数の種", value = "7", change = Change.FIXED) long seed,
      @Key(doc = "割合", value = "0.5", change = Change.COLLECT) @Range(min = 0, max = 1) double rate,
      @Key(doc = "秘密のトークン", presence = Presence.ABSENT, change = Change.FIXED, secret = true) String token) {}

  enum Mode {
    FAST,
    SLOW
  }

  sealed interface DemoCommand extends Command<Demo> permits Train, EvalRun, Boom, Wait, Halt {}

  /** run を動かすコマンド。 */
  @Named(value = "train", doc = "run を回す", reads = Reads.RUN)
  record Train(
      @Key(doc = "run のディレクトリ", presence = Presence.REQUIRED) Path run,
      @Key(doc = "止めるまでに終える反復の数", value = "1") @Range(min = 1) int stopAfter)
      implements DemoCommand {

    @Override
    public Outcome run(Context<Demo> context) {
      context.out().print("train " + run + " " + stopAfter);
      return Outcome.ok("trained");
    }
  }

  /** 2 語の名前で、くり返せる引数を持つコマンド。 */
  @Named(value = "eval run", doc = "卓を測る")
  record EvalRun(
      @Key(doc = "卓", value = "[]") List<String> table,
      @Key(doc = "測り方", value = "FAST") Mode mode,
      @Key(doc = "出力のディレクトリ", presence = Presence.ABSENT) Path out)
      implements DemoCommand {

    @Override
    public Outcome run(Context<Demo> context) {
      Demo settings = context.settings().value();
      context
          .out()
          .print(
              "table=" + table + " mode=" + mode + " out=" + out + " workers=" + settings.workers() + " rate="
                  + settings.rate() + " presets=" + context.presets());
      return Outcome.ok("measured " + table.size());
    }
  }

  @Named(value = "boom", doc = "例外で落ちる")
  record Boom() implements DemoCommand {

    @Override
    public Outcome run(Context<Demo> context) {
      throw new IllegalStateException("boom");
    }
  }

  @Named(value = "wait", doc = "待っている間に割り込まれる")
  record Wait() implements DemoCommand {

    @Override
    public Outcome run(Context<Demo> context) throws InterruptedException {
      throw new InterruptedException("interrupted");
    }
  }

  /** 停止の要求を受けていれば、セーフポイントで止まる。 */
  @Named(value = "halt", doc = "止める頼みがあれば止まる")
  record Halt() implements DemoCommand {

    @Override
    public Outcome run(Context<Demo> context) {
      Signals signals = context.signals();
      return signals.requested() ? new Outcome(Exit.STOPPED, "stopped by " + signals.reason()) : Outcome.ok("ran");
    }
  }

  static Application<Demo> application() {
    List<Class<? extends Command<Demo>>> commands = List.of(DemoCommand.class);
    return Application.of("demo", Demo.class, commands);
  }
}
