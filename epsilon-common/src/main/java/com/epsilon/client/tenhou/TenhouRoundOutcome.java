package com.epsilon.client.tenhou;

import com.epsilon.core.TurnEvent;
import com.epsilon.engine.PostCallDahaiRestriction;

/** 天鳳イベントを局面へ反映した後に、自家が判断する必要のある行動を表す。 */
public sealed interface TenhouRoundOutcome
    permits TenhouRoundOutcome.None,
        TenhouRoundOutcome.DrawDecision,
        TenhouRoundOutcome.ResponseDecision,
        TenhouRoundOutcome.PostCallDahai {

  enum None implements TenhouRoundOutcome {
    INSTANCE
  }

  record DrawDecision(TurnEvent.Draw draw, TenhouEvent.ActionPrompt prompt)
      implements TenhouRoundOutcome {}

  record ResponseDecision(TurnEvent.ResponseSource source, TenhouEvent.ActionPrompt prompt)
      implements TenhouRoundOutcome {}

  record PostCallDahai(PostCallDahaiRestriction restriction) implements TenhouRoundOutcome {}
}
