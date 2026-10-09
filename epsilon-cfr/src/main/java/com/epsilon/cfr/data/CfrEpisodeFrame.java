package com.epsilon.cfr.data;

import java.util.Objects;

/** 半荘内の一つの学習経路。局単位の経路にも半荘全体の経路にも、同じ半荘IDと連番を付ける。 */
public record CfrEpisodeFrame(CfrEpisode episode, int index, boolean last) {
  public CfrEpisodeFrame {
    Objects.requireNonNull(episode, "episode");
    if (index < 0) throw new IllegalArgumentException("Negative CFR trajectory index");
  }
}
