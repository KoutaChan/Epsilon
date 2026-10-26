package com.epsilon.prism.run;

/**
 * run の進み（state.json）。next は次に収集するコホート（= 次に使う版。重みと番号は snapshots/next）、totalRows はこれまでに集めた行の
 * 総数、accepted は蒸留した網を受け入れた反復の数（どちらも報告だけ）、averageWeight は方策の記憶（{@link Averaging}）の power の重みの累計
 * （版 next まで。ほかの決め方では 0）。
 */
public record RunState(int format, int next, long totalRows, int accepted, double averageWeight) {
  public static final int FORMAT = 2;
}
