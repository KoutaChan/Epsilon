package com.epsilon.reviewer;

import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Java HTTP サーバーの起動入口です。 */
public final class Main {
  private static final Logger LOG = LoggerFactory.getLogger(Main.class);

  public static void main(String[] args) throws Exception {
    Map<String, String> options = new HashMap<>();
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("--help")) {
        LOG.info(
            "Epsilon AI Reviewer\n"
                + "--host 127.0.0.1 --port 8080 --origin http://127.0.0.1:8080\n"
                + "--models models.json --data-dir ./reviewer-data --web-dir ../web/dist\n"
                + "Without --models the catalog is empty. Set --origin to the public HTTPS URL when"
                + " publishing.");
        return;
      }
      if (!Set.of("--host", "--port", "--origin", "--models", "--data-dir", "--web-dir")
              .contains(args[i])
          || i + 1 >= args.length)
        throw new IllegalArgumentException("Unknown command-line option: " + args[i]);
      options.put(args[i], args[++i]);
    }
    String host = options.getOrDefault("--host", "127.0.0.1");
    int port = Integer.parseInt(options.getOrDefault("--port", "8080"));
    if (port < 1 || port > 65535)
      throw new IllegalArgumentException("Port must be between 1 and 65535.");
    URI origin = URI.create(options.getOrDefault("--origin", "http://127.0.0.1:" + port));
    if (!Set.of("http", "https").contains(origin.getScheme())
        || origin.getHost() == null
        || origin.getUserInfo() != null
        || origin.getRawQuery() != null
        || origin.getFragment() != null
        || !(origin.getPath().isEmpty() || origin.getPath().equals("/")))
      throw new IllegalArgumentException("Origin must use http(s)://host[:port].");
    origin = URI.create(origin.getScheme() + "://" + origin.getRawAuthority());
    System.setProperty("sun.net.httpserver.maxReqTime", "60");
    System.setProperty("sun.net.httpserver.maxRspTime", "120");
    System.setProperty("sun.net.httpserver.maxConnections", "64");
    Path models = options.containsKey("--models") ? Path.of(options.get("--models")) : null;
    Path web = options.containsKey("--web-dir") ? Path.of(options.get("--web-dir")) : null;
    Path data = Path.of(options.getOrDefault("--data-dir", "reviewer-data"));
    var server = new ReviewerServer(new InetSocketAddress(host, port), origin, models, data, web);
    Runtime.getRuntime().addShutdownHook(new Thread(server::close, "reviewer-shutdown"));
    server.start();
    LOG.info("Epsilon AI Reviewer started at {}.", origin);
    new CountDownLatch(1).await();
  }
}
