package com.epsilon.nano.training;

import com.epsilon.ai.belief.EpsilonBeliefSample;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.training.replay.EpsilonReplaySampleCollector;
import com.epsilon.replay.ReplayRecordReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** 共通の読み込み処理で得た牌譜を、この系列の特徴量と教師データへ変換する。 */
public final class EpsilonLogPretrainDataCollector {
  private EpsilonLogPretrainDataCollector() {}

  public static List<EpsilonDecisionSample> collectDecisionFile(Path file) throws IOException {
    return EpsilonReplaySampleCollector.collectDecisionSamples(ReplayRecordReader.readRecord(file));
  }

  public static List<EpsilonBeliefSample<DecisionHostBatch>> collectBeliefFile(Path file)
      throws IOException {
    return EpsilonReplaySampleCollector.collectBeliefSamples(ReplayRecordReader.readRecord(file));
  }

  public static List<Path> listLogFiles(Path root, int maximumFiles) throws IOException {
    return ReplayRecordReader.listRecordFiles(root, maximumFiles);
  }
}
