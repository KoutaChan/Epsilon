package com.epsilon.reviewer;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.App;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.reviewer.analysis.AnalysisSettings;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * Reviewer を起動する入口を確かめる。
 *
 * <ul>
 *   <li>{@code serve} と {@code engine} の引数と全体の設定は、書かなければ既定値になる。
 *   <li>HTTP のサーバは、範囲外のポートや設定の値、空のアドレス、origin を書かないワイルドカードのアドレス、形の違う origin、
 *       知らない引数や設定のキー、壊れた models.json では起動せず、終了コード 2 で終わる。
 *   <li>デスクトップのエンジンは、環境変数で渡した保存先と models.json を {@code engine} のキーに置き換えて起動する。これは
 *       子プロセスで動かして確かめる。
 * </ul>
 */
public class MainTest {

  private static final String RULES =
      "{\"models\":[{\"modelId\":\"rules\",\"revision\":\"r1\",\"displayName\":\"Rules\",\"version\":\"1\","
          + "\"series\":\"" + Series.RULE_BOT.id() + "\",\"variant\":\"TENHOU_4P\"}]}";

  private static final String QUOTED =
      "{\"models\":[{\"modelId\":\"rules\",\"revision\":\"r1\",\"displayName\":\"Rules\",\"version\":\"1\","
          + "\"series\":\"" + Series.RULE_BOT.id() + "\",\"variant\":\"TENHOU_4P\","
          + "\"options\":{\"temperature\":\"0.004\"}}]}";

  /** 1 回呼び出したときの終了コード、出力、ログの行。 */
  record Ran(int exit, byte[] out, List<String> logs) {

    boolean logged(String text) {
      return logs.stream().anyMatch(line -> line.contains(text));
    }

    @Override
    public String toString() {
      return "exit " + exit + " " + logs;
    }
  }

  private Path directory;

  @BeforeMethod
  public void create() {
    directory = TestFiles.directory("reviewer-main-");
  }

  @AfterMethod(alwaysRun = true)
  public void delete() {
    TestFiles.delete(directory);
  }

  @Test
  public void serveAndTheSettingsBindTheDesignDefaults() {
    Main.Serve serve = section(Main.Serve.class, "serve", "");
    Assert.assertEquals(serve, new Main.Serve("127.0.0.1", 8080, null, null, Path.of("reviewer-data"), null));
    Assert.assertEquals(serve.publicOrigin(), URI.create("http://127.0.0.1:8080"));
    Assert.assertEquals(
        Settings.bind(ReviewerSettings.class, List.of()).value(),
        new ReviewerSettings(new ReviewerSettings.Server(2, 60, 120, 64), new AnalysisSettings(1, 16, 24)));
    Assert.assertEquals(
        section(Main.Engine.class, "engine", "dataDir = \"/d\"\nmodels = \"/m.json\""),
        new Main.Engine(Path.of("/d"), Path.of("/m.json")));
    Assert.assertThrows(SettingsException.class, () -> section(Main.Engine.class, "engine", "dataDir = \"/d\""));
  }

  @Test
  public void thePublicOriginFollowsTheHostOrTheWrittenOrigin() {
    Assert.assertEquals(
        section(Main.Serve.class, "serve", "host = \"::1\"\nport = 8097").publicOrigin(),
        URI.create("http://[::1]:8097"));
    Assert.assertEquals(
        section(Main.Serve.class, "serve", "host = \"0.0.0.0\"\norigin = \"https://reviewer.test/\"").publicOrigin(),
        URI.create("https://reviewer.test"));
    Assert.assertEquals(
        section(Main.Serve.class, "serve", "origin = \"http://reviewer.test:8097\"").publicOrigin(),
        URI.create("http://reviewer.test:8097"));
  }

  @Test
  public void invalidServeArgumentsExitWithUsageBeforeServing() throws IOException {
    Path data = directory.resolve("data");
    Path broken = Files.writeString(directory.resolve("models.json"), QUOTED);
    String[][] invalid = {
      {"serve", "--port", "0"},
      {"serve", "--port", "70000"},
      {"serve", "--port", "http"},
      {"serve", "--host", " "},
      {"serve", "--host", "0.0.0.0"},
      {"serve", "--host", "::"},
      {"serve", "--origin", "ftp://reviewer.test"},
      {"serve", "--origin", "http://reviewer.test/path"},
      {"serve", "--origin", "http://user@reviewer.test"},
      {"serve", "--origin", "http://reviewer.test/?q=1"},
      {"serve", "--prot", "8097"},
      {"serve", "--set", "analysis.workers=0"},
      {"serve", "--set", "server.maxOpenModels=0"},
      {"serve", "--set", "server.maxConection=8"},
      {"serve", "--port", "8097", "--data-dir", data.toString(), "--models", broken.toString()},
    };
    for (String[] arguments : invalid) {
      Ran ran = run(arguments);
      Assert.assertEquals(ran.exit(), 2, Arrays.toString(arguments) + " " + ran);
    }
    Assert.assertTrue(run(invalid[4]).logged("is required when host is the wildcard address 0.0.0.0"));
    Assert.assertTrue(run(invalid[13]).logged("did you mean server.maxConnections?"));
    Ran quoted = run(invalid[14]);
    Assert.assertTrue(quoted.logged("models[0].options.temperature=0.004"), quoted.toString());
    Assert.assertTrue(quoted.logged("(written as a string: drop the quotes)"), quoted.toString());
    Ran help = run("help", "serve");
    Assert.assertEquals(help.exit(), 0, help.toString());
    String usage = new String(help.out(), StandardCharsets.UTF_8);
    for (String expected :
        List.of(
            "--host", "default 127.0.0.1", "--port", "default 8080", "--origin", "--models", "--data-dir", "--web-dir")) {
      Assert.assertTrue(usage.contains(expected), expected + " in " + usage);
    }
  }

  /**
   * engine は起動できなければ failed のフレームを書き、終了コード 1 で終わる（標準入力は読まない）。必須のキーが無ければ、引数の誤りとして
   * 2 で終わる。
   */
  @Test
  public void anEngineThatCannotStartWritesAFailedFrame() throws IOException {
    Path broken = Files.writeString(directory.resolve("models.json"), QUOTED);
    Ran ran =
        run("engine", "--set", "engine.dataDir=\"" + directory.resolve("data") + "\"", "--set",
            "engine.models=\"" + broken + "\"");
    Assert.assertEquals(ran.exit(), 1, ran.toString());
    List<JsonObject> frames = frames(ran.out());
    Assert.assertEquals(frames.size(), 1, frames.toString());
    Assert.assertEquals(frames.getFirst().get("event").getAsString(), "failed");
    Assert.assertTrue(frames.getFirst().toString().contains("drop the quotes"), frames.toString());
    Ran missing = run("engine", "--set", "engine.dataDir=\"" + directory.resolve("data") + "\"");
    Assert.assertEquals(missing.exit(), 2, missing.toString());
  }

  /**
   * デスクトップのエンジンを子プロセスで動かす。エンジンは、環境変数で渡した保存先と models.json を engine のキーに置き換えて起動し、
   * ready のフレームを書き、標準入力が終わると終了コード 0 で終わる。空白や引用符を含むパスも、そのまま渡る。環境変数が足りなければ、
   * どれが無いかを書いた failed のフレームを書いて 1 で終わる。models.json が壊れていれば、エラーメッセージを書いた failed のフレームを
   * 書いて 1 で終わる。
   */
  @Test(timeOut = 300_000)
  public void theDesktopEngineMapsItsEnvironmentToTheEngineKeys() throws Exception {
    Path data = directory.resolve("da ta \"q\"");
    Path logs = Files.createDirectories(directory.resolve("logs"));
    Path models = Files.writeString(directory.resolve("models \"x\".json"), RULES);
    Child ready =
        child(
            Map.of(
                "EPSILON_REVIEWER_DATA", data.toString(),
                "EPSILON_REVIEWER_MODELS", models.toString(),
                "EPSILON_REVIEWER_LOGS", logs.toString()));
    Assert.assertEquals(ready.exit(), 0, ready.toString());
    Assert.assertEquals(ready.frames().size(), 1, ready.toString());
    Assert.assertEquals(ready.frames().getFirst().get("event").getAsString(), "ready");
    Assert.assertTrue(Files.isDirectory(data), "the data directory " + data + " was not opened");

    Child unset = child(Map.of("EPSILON_REVIEWER_DATA", data.toString(), "EPSILON_REVIEWER_LOGS", logs.toString()));
    Assert.assertEquals(unset.exit(), 1, unset.toString());
    Assert.assertEquals(unset.frames().getFirst().get("event").getAsString(), "failed");
    Assert.assertTrue(
        unset.frames().getFirst().toString().contains("EPSILON_REVIEWER_MODELS is not set."), unset.toString());

    Path broken = Files.writeString(directory.resolve("broken.json"), QUOTED);
    Child invalid =
        child(
            Map.of(
                "EPSILON_REVIEWER_DATA", data.toString(),
                "EPSILON_REVIEWER_MODELS", broken.toString(),
                "EPSILON_REVIEWER_LOGS", logs.toString()));
    Assert.assertEquals(invalid.exit(), 1, invalid.toString());
    Assert.assertEquals(invalid.frames().getFirst().get("event").getAsString(), "failed");
    Assert.assertTrue(invalid.frames().getFirst().toString().contains("drop the quotes"), invalid.toString());
  }

  /** 子プロセスの終了コード、標準出力に書かれたフレーム、標準エラー。 */
  record Child(int exit, List<JsonObject> frames, String errors) {}

  /** StdioMain を子プロセスで起動し、標準入力をすぐ閉じる。 */
  private Child child(Map<String, String> environment) throws Exception {
    Path java = Path.of(System.getProperty("java.home"), "bin", "java");
    ProcessBuilder builder =
        new ProcessBuilder(
            java.toString(), "-Xmx512m", "-cp", System.getProperty("java.class.path"), StdioMain.class.getName());
    Map<String, String> env = builder.environment();
    env.remove("EPSILON_REVIEWER_DATA");
    env.remove("EPSILON_REVIEWER_MODELS");
    env.remove("EPSILON_REVIEWER_LOGS");
    env.putAll(environment);
    Path errors = directory.resolve("child-" + System.nanoTime() + ".err");
    builder.redirectError(errors.toFile());
    Process process = builder.start();
    process.getOutputStream().close();
    byte[] out = process.getInputStream().readAllBytes();
    if (!process.waitFor(120, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      Assert.fail("the engine did not end: " + Files.readString(errors));
    }
    return new Child(process.exitValue(), frames(out), Files.readString(errors));
  }

  /** フレームの並び（ヘッダーの長さ・ヘッダーの JSON・本文の長さ・本文）から、ヘッダーだけを取り出す。 */
  private static List<JsonObject> frames(byte[] bytes) throws IOException {
    List<JsonObject> frames = new ArrayList<>();
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      while (true) {
        int length;
        try {
          length = in.readInt();
        } catch (EOFException end) {
          return frames;
        }
        byte[] header = in.readNBytes(length);
        frames.add(JsonParser.parseString(new String(header, StandardCharsets.UTF_8)).getAsJsonObject());
        in.skipNBytes(in.readInt());
      }
    }
  }

  private static <T extends Record> T section(Class<T> type, String path, String body) {
    String text = "[" + path + "]\n" + body + "\n";
    return Settings.section(type, path, List.of(Layer.parse("reviewer.toml", Layer.Source.FILE, text))).value();
  }

  private static Ran run(String... args) {
    Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
    ListAppender<ILoggingEvent> appender = new ListAppender<>();
    appender.start();
    root.addAppender(appender);
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    int exit;
    try (PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
      exit = App.run(Main.application(), args, out);
    } finally {
      root.detachAppender(appender);
    }
    List<String> logs = appender.list.stream().map(ILoggingEvent::getFormattedMessage).toList();
    return new Ran(exit, bytes.toByteArray(), logs);
  }
}
