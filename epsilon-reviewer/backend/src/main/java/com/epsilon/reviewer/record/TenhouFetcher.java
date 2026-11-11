package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.zip.GZIPInputStream;

/**
 * 天鳳から牌譜を取ってくる。接続するのは天鳳のホストだけで、リダイレクトは同じホストとポートへのものだけを 3 回まで追う。DNS で引いた
 * アドレスが公開のアドレス（ループバック・リンクローカル・プライベート・マルチキャストでない）であることも確かめる。gzip での
 * 転送を求め、届いた本文を展開する。本文は受け取る途中でも 16 MiB を超えたら止め、1 回の要求は 30 秒で打ち切る。
 */
public final class TenhouFetcher implements AutoCloseable {
  private static final int LIMIT = RecordImporter.RECORD_LIMIT;
  private static final int MAX_REDIRECTS = 3;
  private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
  private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(30);
  private static final Duration DEADLINE = Duration.ofSeconds(30);
  private static final String USER_AGENT = "EpsilonReviewer/1.0";

  private final URI origin;
  private final boolean publicAddressesOnly;
  private final HttpClient client =
      HttpClient.newBuilder()
          .connectTimeout(CONNECT_TIMEOUT)
          .followRedirects(HttpClient.Redirect.NEVER)
          .build();

  public TenhouFetcher() {
    this(URI.create("https://tenhou.net"), true);
  }

  /**
   * @param origin 牌譜を取るホスト（scheme://host[:port]）。テストでは手元の HTTP サーバを渡す
   * @param publicAddressesOnly DNS で引いたアドレスが公開のアドレスであることを確かめるなら true
   */
  TenhouFetcher(URI origin, boolean publicAddressesOnly) {
    this.origin = origin;
    this.publicAddressesOnly = publicAddressesOnly;
  }

  /**
   * @throws ReviewerException record_not_found（404）・record_access_denied（401・403）・fetch_failed
   *     （リダイレクト・DNS・TLS・接続先・その他のステータス）・fetch_timeout・record_too_large
   */
  public byte[] fetch(TenhouLogId id) {
    URI download = id.downloadUri();
    URI uri = origin.resolve(download.getRawPath() + "?" + download.getRawQuery());
    try {
      for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
        requireEndpoint(uri);
        HttpResponse<byte[]> response = send(uri);
        int status = response.statusCode();
        if (status >= 300 && status < 400) {
          String location =
              response
                  .headers()
                  .firstValue("Location")
                  .orElseThrow(() -> failed("The record response has no redirect destination."));
          URI next = uri.resolve(location);
          if (!Objects.equals(uri.getHost(), next.getHost()) || uri.getPort() != next.getPort())
            throw failed("A redirect outside the record service was rejected.");
          uri = next;
          continue;
        }
        if (status == 401 || status == 403)
          throw new ReviewerException(ErrorCode.RECORD_ACCESS_DENIED);
        if (status == 404) throw new ReviewerException(ErrorCode.RECORD_NOT_FOUND);
        if (status != 200) throw failed("The record service returned status " + status + ".");
        return decode(response);
      }
      throw failed("The record service redirected too many times.");
    } catch (IOException | IllegalArgumentException failure) {
      throw failed(failure.toString());
    }
  }

  @Override
  public void close() {
    client.close();
  }

  private void requireEndpoint(URI uri) throws IOException {
    boolean allowed =
        origin.getScheme().equals(uri.getScheme())
            && uri.getUserInfo() == null
            && origin.getHost().equals(uri.getHost())
            && origin.getPort() == uri.getPort();
    if (!allowed) throw failed("The record fetch destination is not allowed.");
    if (!publicAddressesOnly) return;
    for (InetAddress address : InetAddress.getAllByName(uri.getHost()))
      if (address.isAnyLocalAddress()
          || address.isLoopbackAddress()
          || address.isLinkLocalAddress()
          || address.isSiteLocalAddress()
          || address.isMulticastAddress())
        throw failed("The record service resolved to a disallowed address.");
  }

  private HttpResponse<byte[]> send(URI uri) {
    HttpRequest request =
        HttpRequest.newBuilder(uri)
            .timeout(REQUEST_TIMEOUT)
            .header("User-Agent", USER_AGENT)
            .header("Accept-Encoding", "gzip")
            .GET()
            .build();
    CompletableFuture<HttpResponse<byte[]>> pending =
        client.sendAsync(request, info -> new BoundedBody(LIMIT));
    try {
      return pending.get(DEADLINE.toMillis(), TimeUnit.MILLISECONDS);
    } catch (TimeoutException failure) {
      throw new ReviewerException(ErrorCode.FETCH_TIMEOUT);
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new ReviewerException(ErrorCode.SHUTTING_DOWN);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof ReviewerException reviewer) throw reviewer;
      if (cause instanceof HttpTimeoutException)
        throw new ReviewerException(ErrorCode.FETCH_TIMEOUT);
      throw failed(String.valueOf(cause));
    } finally {
      if (!pending.isDone()) pending.cancel(true);
    }
  }

  /** Content-Encoding が gzip なら 16 MiB まで展開する。 */
  private static byte[] decode(HttpResponse<byte[]> response) throws IOException {
    boolean gzip =
        response
            .headers()
            .firstValue("Content-Encoding")
            .map(encoding -> encoding.trim().equalsIgnoreCase("gzip"))
            .orElse(false);
    if (!gzip) return response.body();
    try (var in = new GZIPInputStream(new ByteArrayInputStream(response.body()))) {
      byte[] inflated = in.readNBytes(LIMIT + 1);
      if (inflated.length > LIMIT) throw new ReviewerException(ErrorCode.RECORD_TOO_LARGE);
      return inflated;
    }
  }

  private static ReviewerException failed(String message) {
    return new ReviewerException(ErrorCode.FETCH_FAILED, message);
  }

  /** 本文を受け取り、途中で上限を超えたら受け取りを止める。 */
  private static final class BoundedBody implements HttpResponse.BodySubscriber<byte[]> {
    private final int limit;
    private final ByteArrayOutputStream received = new ByteArrayOutputStream();
    private final CompletableFuture<byte[]> body = new CompletableFuture<>();
    private Flow.Subscription subscription;

    BoundedBody(int limit) {
      this.limit = limit;
    }

    @Override
    public CompletionStage<byte[]> getBody() {
      return body;
    }

    @Override
    public void onSubscribe(Flow.Subscription subscription) {
      this.subscription = subscription;
      subscription.request(1);
    }

    @Override
    public void onNext(List<ByteBuffer> buffers) {
      if (body.isDone()) return;
      for (ByteBuffer buffer : buffers) {
        if (buffer.remaining() > limit - received.size()) {
          subscription.cancel();
          body.completeExceptionally(new ReviewerException(ErrorCode.RECORD_TOO_LARGE));
          return;
        }
        if (buffer.hasArray()) {
          received.write(
              buffer.array(), buffer.arrayOffset() + buffer.position(), buffer.remaining());
        } else {
          byte[] chunk = new byte[buffer.remaining()];
          buffer.get(chunk);
          received.write(chunk, 0, chunk.length);
        }
      }
      subscription.request(1);
    }

    @Override
    public void onError(Throwable failure) {
      body.completeExceptionally(failure);
    }

    @Override
    public void onComplete() {
      body.complete(received.toByteArray());
    }
  }
}
