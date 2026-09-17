package com.epsilon.nano.ai.decision.data;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;

/** 一つの学習データ片から選抜済みサンプルの参照情報を読み、独立したリストへまとめる。 */
@FunctionalInterface
public interface EpsilonDecisionTrainingSampleDescriptorReader {

  ArrayList<EpsilonDecisionTrainingSampleDescriptor> read(Path path) throws IOException;
}
