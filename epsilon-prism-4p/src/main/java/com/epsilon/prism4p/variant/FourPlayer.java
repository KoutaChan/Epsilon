package com.epsilon.prism4p.variant;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import com.epsilon.agent.app.Command;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.eval.OpponentRules;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.AuxiliaryTargets;
import com.epsilon.prism.learn.TargetWriter;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.variant.PrismVariant;
import com.epsilon.prism4p.eval.FourPlayerOpponents;
import java.util.List;

/** 天鳳の四人打ちの PRISM のモジュール。四人打ちだけのコマンドは持たない。 */
public final class FourPlayer implements PrismVariant {
  @Override
  public Variant variant() {
    return Variant.TENHOU_4P;
  }

  @Override
  public InputFormat input() {
    return FOUR_PLAYER;
  }

  @Override
  public TargetWriter targets(AuxiliaryLoss loss) {
    return new AuxiliaryTargets(loss, Variant.TENHOU_4P);
  }

  @Override
  public OpponentRules opponents() {
    return FourPlayerOpponents.RULES;
  }

  @Override
  public List<Class<? extends Command<Record>>> commands() {
    return List.of();
  }
}
