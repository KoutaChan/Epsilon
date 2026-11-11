package com.epsilon.prism.eval;

import com.epsilon.agent.policy.Series;
import java.util.List;

/**
 * 評価の卓が受け付ける相手の系列と、チェックポイントを開く相手の既定のデバイスを、四人打ち・三人打ちごとに決める。PRISM のバージョンは、
 * どちらでも相手に書ける。
 */
public interface OpponentRules {

  /** PRISM のバージョンのほかに、相手に書ける系列（指定の説明に並べる順）。 */
  List<Series> series();

  /** チェックポイントを開く相手の既定のデバイス（cpu か gpuN）。 */
  String defaultDevice();
}
