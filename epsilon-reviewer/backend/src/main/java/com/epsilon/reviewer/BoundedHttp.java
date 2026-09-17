package com.epsilon.reviewer;

import java.io.*;
import java.net.http.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.Flow;

/** HTTP 本文の受信中にも容量と時間の上限を適用します。 */
final class BoundedHttp {
  static HttpResponse<byte[]> get(HttpClient client, HttpRequest request, int maximum)
      throws IOException, InterruptedException {
    var pending = client.sendAsync(request, info -> new Body(maximum));
    try {
      return pending.get(30, TimeUnit.SECONDS);
    } catch (TimeoutException e) {
      throw new ApiException(504, "fetch_timeout", "Record service response timed out.");
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof ApiException api) throw api;
      if (cause instanceof IOException io) throw io;
      throw new IOException("Could not receive the record.", cause);
    } finally {
      if (!pending.isDone()) pending.cancel(true);
    }
  }

  private static final class Body implements HttpResponse.BodySubscriber<byte[]> {
    private final int maximum;
    private final ByteArrayOutputStream received = new ByteArrayOutputStream();
    private final CompletableFuture<byte[]> completed = new CompletableFuture<>();
    private Flow.Subscription subscription;

    Body(int maximum) {
      this.maximum = maximum;
    }

    @Override
    public CompletionStage<byte[]> getBody() {
      return completed;
    }

    @Override
    public void onSubscribe(Flow.Subscription subscription) {
      this.subscription = subscription;
      subscription.request(1);
    }

    @Override
    public void onNext(List<ByteBuffer> buffers) {
      for (ByteBuffer buffer : buffers) {
        if (buffer.remaining() > maximum - received.size()) {
          subscription.cancel();
          completed.completeExceptionally(
              new ApiException(413, "record_too_large", "Record data exceeds the size limit."));
          return;
        }
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        received.writeBytes(bytes);
      }
      subscription.request(1);
    }

    @Override
    public void onError(Throwable throwable) {
      completed.completeExceptionally(throwable);
    }

    @Override
    public void onComplete() {
      completed.complete(received.toByteArray());
    }
  }
}
