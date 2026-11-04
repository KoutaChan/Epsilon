package com.epsilon.reviewer;

import com.epsilon.reviewer.http.HttpReviewer;
import com.epsilon.reviewer.http.ServerConfig;
import com.epsilon.reviewer.http.ServerOption;
import java.io.IOException;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** HTTP のサーバの CLI。 */
public final class Main {
  private static final Logger LOG = LoggerFactory.getLogger(Main.class);

  private static final int MAX_OPEN_MODELS = 2;

  private Main() {}

  public static void main(String[] args) throws IOException {
    if (List.of(args).contains(ServerOption.HELP.flag())) {
      LOG.info(ServerOption.usage());
      return;
    }
    ServerConfig config = ServerConfig.parse(args);
    // JDK の HttpServer は最初の作成の時にこの値を読む。
    System.setProperty("sun.net.httpserver.maxReqTime", "60");
    System.setProperty("sun.net.httpserver.maxRspTime", "120");
    System.setProperty("sun.net.httpserver.maxConnections", "64");
    Reviewer reviewer =
        Reviewer.open(new ReviewerConfig(config.dataDir(), config.models(), MAX_OPEN_MODELS));
    HttpReviewer server;
    try {
      server = HttpReviewer.start(config, reviewer);
    } catch (IOException | RuntimeException failure) {
      reviewer.close();
      throw failure;
    }
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  server.close();
                  reviewer.close();
                },
                "reviewer-shutdown"));
    LOG.info("Epsilon AI Reviewer started at {} (port {}).", config.origin(), server.port());
  }
}
