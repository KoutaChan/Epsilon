package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.prism.TableWalk;
import java.util.EnumSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 受付の区分をテストする。合法手にチーかポンがあれば {@code MULTIPLE_TRANSITIONS}、無ければ {@code SINGLE_TRANSITION} になることを、鳴きの多い半荘の全ての判断で確かめる。 */
public class CapacityTest {

  @Test
  public void callsNeedTheLargerCapacity() {
    Observation o = new Observation(TENHOU_4P);
    EnumSet<Capacity> seen = EnumSet.noneOf(Capacity.class);
    for (long[] words : PolicyRequests.observations(5, TableWalk.Policy.CALLS, Integer.MAX_VALUE)) {
      o.bind(words, 0);
      boolean call = false;
      for (int a = 0; a < o.legalActions(); a++) {
        Action.Type type = o.action(a).type();
        call |= type == Action.Type.CHI || type == Action.Type.PON;
      }
      Capacity capacity = Capacity.of(o);
      Assert.assertEquals(capacity, call ? Capacity.MULTIPLE_TRANSITIONS : Capacity.SINGLE_TRANSITION);
      seen.add(capacity);
    }
    Assert.assertEquals(seen, EnumSet.allOf(Capacity.class));
  }
}
