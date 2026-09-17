package com.epsilon.arena;

import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.ModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 複数席で使うモデルを一度だけ開き、終了時に一部の解放が失敗しても残りを解放することを検証する。 */
public class ModelRegistryTest {
  @Test
  public void oneCloseFailureDoesNotLeaveTheOtherModelsOpen() throws Exception {
    Path checkpoint = Files.createTempFile("arena-close", ".params");
    AtomicInteger opened = new AtomicInteger();
    AtomicInteger closed = new AtomicInteger();
    ModelProvider provider =
        new ModelProvider() {
          public String seriesId() {
            return "test";
          }

          public BatchedPolicy open(
              Path path, Map<String, String> settings, PolicyExecutionContext context) {
            int index = opened.getAndIncrement();
            return new BatchedPolicy() {
              public Object batchKey(DecisionRequest request) {
                throw new UnsupportedOperationException();
              }

              public int maxBatchSize(Object key) {
                throw new UnsupportedOperationException();
              }

              public Ingress tryAcquire(Object key, Runnable wakeup) {
                throw new UnsupportedOperationException();
              }

              public void close() {
                closed.incrementAndGet();
                if (index == 2) throw new IllegalStateException("last model close failed");
                if (index == 1) throw new AssertionError("middle model close failed");
              }
            };
          }
        };
    try (PolicyExecutionContext context = new PolicyExecutionContext()) {
      ModelRegistry registry = new ModelRegistry(List.of(provider), context);
      try {
        for (int index = 0; index < 3; index++)
          registry.open(
              new ModelSpec("test", checkpoint, Map.of("instance", Integer.toString(index))));
        IllegalStateException error =
            Assert.expectThrows(IllegalStateException.class, registry::close);
        Assert.assertEquals(closed.get(), 3);
        Assert.assertEquals(error.getSuppressed().length, 1);
        Assert.assertTrue(error.getSuppressed()[0] instanceof AssertionError);
        registry.close();
        Assert.assertEquals(closed.get(), 3);
      } finally {
        registry.close();
      }
    } finally {
      Files.delete(checkpoint);
    }
  }

  @Test
  public void sameModelAcrossSeatsOpensAndClosesOnlyOnce() throws Exception {
    Path checkpoint = Files.createTempFile("arena-checkpoint", ".params");
    AtomicInteger opens = new AtomicInteger();
    AtomicInteger closes = new AtomicInteger();
    ModelProvider provider =
        new ModelProvider() {
          public String seriesId() {
            return "test";
          }

          public BatchedPolicy open(
              Path path, Map<String, String> settings, PolicyExecutionContext execution) {
            opens.incrementAndGet();
            return new BatchedPolicy() {
              public Object batchKey(DecisionRequest request) {
                throw new UnsupportedOperationException();
              }

              public int maxBatchSize(Object key) {
                throw new UnsupportedOperationException();
              }

              public Ingress tryAcquire(Object key, Runnable wakeup) {
                throw new UnsupportedOperationException();
              }

              public void close() {
                closes.incrementAndGet();
              }
            };
          }
        };
    try (PolicyExecutionContext context = new PolicyExecutionContext()) {
      try (ModelRegistry registry = new ModelRegistry(List.of(provider), context)) {
        ModelSpec spec = new ModelSpec("test", checkpoint, Map.of("device", "cpu"));
        Participant first = registry.open(spec);
        Assert.assertSame(first.policy(), registry.open(spec).policy());
        Assert.assertSame(first.policy(), registry.open(spec).policy());
        Assert.assertEquals(opens.get(), 1);
        Participant different =
            registry.open(new ModelSpec("test", checkpoint, Map.of("device", "gpu:0")));
        Assert.assertNotSame(first.policy(), different.policy());
        Assert.assertEquals(opens.get(), 2);
      }
      Assert.assertEquals(closes.get(), 2);
    } finally {
      Files.delete(checkpoint);
    }
  }
}
