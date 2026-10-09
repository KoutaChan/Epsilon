package com.epsilon.cfr.collection;

import com.epsilon.cfr.data.CfrEpisodeFrame;
import com.epsilon.cfr.data.CfrEpisodeWriter;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;

/** 完了した対局を、収集処理を待機させずに受け付ける。 */
public interface CfrEpisodeSink {
  /** 受け付けた場合だけ{@code true}を返す。{@code false}の場合は対局を保持せず、収集側に所有権を残す。 このメソッドでは空き待ちを行わない。 */
  boolean tryAccept(CfrCollector.CollectedEpisode episode);

  /**
   * 受け付け可能になったときに完了する通知を返す。既に空きがあれば完了済みの通知を返し、失敗した場合は例外で完了する。 通知は枠の予約ではないため、受信後も{@link
   * #tryAccept}で受け付けを確認する。
   */
  CompletionStage<Void> whenAvailable();

  /** 保存先の空き通知を収集処理へ接続する。保存の完了確認とワーカーの終了は呼び出し側の{@code writer.close()}が担う。 */
  static CfrEpisodeSink persisting(CfrEpisodeWriter writer) {
    Objects.requireNonNull(writer);
    return new CfrEpisodeSink() {
      @Override
      public boolean tryAccept(CfrCollector.CollectedEpisode episode) {
        return writer.tryAccept(
            new CfrEpisodeFrame(episode.episode(), episode.trajectoryIndex(), episode.gameEnded()));
      }

      @Override
      public CompletionStage<Void> whenAvailable() {
        return writer.whenAvailable();
      }
    };
  }

  /** 待機を必要としない処理へ完了した対局を渡す。 */
  static CfrEpisodeSink consuming(Consumer<CfrCollector.CollectedEpisode> consumer) {
    Objects.requireNonNull(consumer);
    return new CfrEpisodeSink() {
      @Override
      public boolean tryAccept(CfrCollector.CollectedEpisode episode) {
        consumer.accept(episode);
        return true;
      }

      @Override
      public CompletionStage<Void> whenAvailable() {
        return CompletableFuture.completedFuture(null);
      }
    };
  }
}
