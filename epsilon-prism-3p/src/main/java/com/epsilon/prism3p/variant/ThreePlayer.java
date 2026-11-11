package com.epsilon.prism3p.variant;

import com.epsilon.agent.app.Command;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.eval.OpponentRules;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.AuxiliaryTargets;
import com.epsilon.prism.learn.TargetWriter;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.variant.PrismVariant;
import com.epsilon.prism3p.eval.ThreePlayerOpponents;
import com.epsilon.prism3p.input.InputSchema;
import java.util.List;

/** 天鳳の三人打ちの PRISM のモジュール。三人打ちだけのコマンドは持たない。 */
public final class ThreePlayer implements PrismVariant {

  private static final Variant VARIANT = Variant.TENHOU_3P;

  @Override
  public Variant variant() {
    return VARIANT;
  }

  @Override
  public InputFormat input() {
    return InputSchema.FORMAT;
  }

  @Override
  public TargetWriter targets(AuxiliaryLoss loss) {
    return new AuxiliaryTargets(loss, VARIANT);
  }

  @Override
  public OpponentRules opponents() {
    return ThreePlayerOpponents.RULES;
  }

  @Override
  public List<Class<? extends Command<Record>>> commands() {
    return List.of();
  }
}
