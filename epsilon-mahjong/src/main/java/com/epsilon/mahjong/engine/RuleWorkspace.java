package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import java.util.Objects;

/**
 * ルールの計算に使う作業領域（形の解析と、採点の入力と結果）。1 つのスレッドだけで使う。対局は判断の時点ごとに別のスレッドへ
 * 移ることがある。そのためエンジンは作業領域を持たず、局を進める呼び出しと読む呼び出しで引数として受け取る。
 */
public final class RuleWorkspace {

  private final Variant variant;
  private final ShapeAnalyzer analyzer;
  private final HandScorer scorer = new HandScorer();
  private final WinContext context;
  private final WinScore score = new WinScore();

  /** 打牌や暗槓の後の形を、手を変えずに試すためのコピー。合法手を作るときと、局末の待ちを求めるときに使う。 */
  private final HandShape probe = new HandShape();

  public RuleWorkspace(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    analyzer = new ShapeAnalyzer(variant);
    context = new WinContext(variant);
  }

  Variant variant() {
    return variant;
  }

  ShapeAnalyzer analyzer() {
    return analyzer;
  }

  HandScorer scorer() {
    return scorer;
  }

  WinContext context() {
    return context;
  }

  WinScore score() {
    return score;
  }

  HandShape probe() {
    return probe;
  }
}
