package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.HipParity;
import com.epsilon.prism.backend.PerturbedNetwork;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.kernel.CandidateKernels;
import com.epsilon.prism.kernel.RelationAttention;
import com.epsilon.prism.kernel.RelationCodes;
import com.epsilon.prism.kernel.TrainingKernels;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.BiFunction;
import org.testng.Assert;

/** 固定した入力で学習し、各演算と更新結果を呼び出し順に記録する。 */
final class TrainingStepRecorder {
  private TrainingStepRecorder() {}

  interface Calls {
    void save(String name, Object[] arguments) throws IOException;
  }

  static void record(
      Path path, String arithmetic, BiFunction<PrismNetwork, PtStream, TrainingKernels> open)
      throws IOException, MalformedModelException {
    record(path, arithmetic, Device.gpu(0), open, false, null);
  }

  static void record(
      Path path,
      String arithmetic,
      Device device,
      BiFunction<PrismNetwork, PtStream, TrainingKernels> open,
      boolean endpoints,
      Calls calls)
      throws IOException, MalformedModelException {
    try (var trace = new TrainingTensorTrace(path);
        var manager = NDManager.newBaseManager(device, "PyTorch");
        var stream = device.isGpu() ? ((PtEngine) manager.getEngine()).newStream(device) : null;
        var scope = stream == null ? null : stream.openScope()) {
      var network =
          PerturbedNetwork.of(manager, InputSchema.FOUR_PLAYER, ShapeArm.V4_M.shape(), 229);
      String environment =
          arithmetic + manager.getEngine().getVersion() + "/BF16/" + ShapeArm.V4_M.name();
      for (var param : network.getParameters()) {
        var weight = param.getValue().getArray();
        weight.setRequiresGradient(true);
        trace.tensor("start/" + environment + "/" + param.getKey(), weight);
      }
      Set<String> called = new HashSet<>();
      var selected = open == null ? null : open.apply(network, stream);
      var kernels =
          selected == null ? null : recording(selected, endpoints ? null : trace, called, calls);
      Optimizer adam =
          Optimizer.adamW()
              .optLearningRateTracker(Tracker.fixed(1e-4f))
              .optWeightDecays(0f)
              .build();
      for (int step = 0; step < 3; step++) {
        var rows = HipFixtures.mixedRows(8, 800 + step);
        var host = HipFixtures.encode(rows, network.shape());
        var actions = HipParity.Actions.of(host);
        List<ValueLabel> labels = new ArrayList<>();
        Random random = new Random(11 + step);
        for (int n : actions.rowCount()) {
          float[] values = new float[n];
          for (int i = 0; i < n; i++) values[i] = random.nextFloat() - 0.5f;
          labels.add(new ValueLabel(values, 0.1f, 1f));
        }
        var aux =
            Batches.auxiliary(
                labels.size(),
                actions.roles().length,
                a -> TargetSchema.hasCandidateTargets(actions.roles()[a]),
                13 + step);
        try (var work = manager.newSubManager();
            var gc = manager.getEngine().newGradientCollector()) {
          var tape = kernels == null ? null : new AttentionTape(kernels);
          var inputs = EagerStages.inputs(work, host);
          var pass = Pass.training(Precision.BFLOAT16, ValueLabel.measured(labels), aux, tape);
          if (endpoints) {
            for (int r = 0; r < rows.size(); r++) {
              // 局面の全ビットを正確に比較できるよう、各ワードを 4 個の符号なし 16 ビット整数に分ける。
              long[] words = rows.get(r);
              int[] pieces = new int[4 * words.length];
              for (int i = 0; i < pieces.length; i++)
                pieces[i] = (int) (words[i / 4] >>> (16 * (i % 4))) & 65535;
              trace.tensor(step + "/input/row/" + r, work.create(pieces));
              trace.tensor(step + "/input/label/" + r, work.create(labels.get(r).values()));
            }
            trace.tensor(step + "/input/actions", work.create(aux.candidateActions()));
            trace.tensor(step + "/input/outcomes", work.create(aux.outcomes()));
            trace.tensor(step + "/input/candidateTargets", work.create(aux.candidateValues()));
            trace.tensor(step + "/input/readingRows", work.create(aux.readingRows()));
            trace.tensor(step + "/input/readingTargets", work.create(aux.readingValues()));
            if (kernels == null) {
              var index = inputs.index(pass);
              trace.tensor(step + "/eagerCodes", RelationClass.codes(inputs, index, index.tokens));
            }
          }
          var outputs =
              network.forward(
                  new ParameterStore(manager, false), inputs, pass, Context.Intermediates.CLOSE);
          var loss =
              new PrismLoss(AuxiliaryLoss.standard())
                  .total(work, outputs, labels, aux, 1f, labels.size(), 1f);
          trace.tensor(step + "/scores", outputs.scores());
          if (endpoints) {
            trace.tensor(step + "/candidateHead", outputs.auxiliary().candidates());
            trace.tensor(step + "/readingHead", outputs.auxiliary().reading());
          }
          trace.tensor(step + "/loss", loss);
          if (tape == null) gc.backward(loss);
          else tape.backward(gc, loss);
        }
        for (var param : network.getParameters()) {
          var weight = param.getValue().getArray();
          try (var grad = JniUtils.getGradient((PtNDArray) weight)) {
            Assert.assertNotNull(grad, param.getKey() + " gradient is missing");
            Assert.assertEquals(grad.getShape(), weight.getShape(), param.getKey());
            trace.tensor(step + "/gradient/" + param.getKey(), grad);
            adam.update(param.getValue().getId(), weight, grad);
            trace.tensor(step + "/weight/" + param.getKey(), weight);
          }
          if (step < 2) resetGradient(param.getValue(), manager);
        }
      }
      if (kernels != null)
        Assert.assertEquals(
            called,
            Set.of(
                "codes",
                "relation/forward",
                "relation/backward",
                "causal/forward",
                "causal/backward",
                "norm/forward",
                "norm/backward",
                "count",
                "countGradient",
                "candidateAttention",
                "candidateAttentionGradient"));
    }
  }

  /** 同じ重みとパラメーター ID を保ち、次の逆伝播が書くまで勾配を未定義に戻す。 */
  static void resetGradient(Parameter parameter, NDManager manager)
      throws IOException, MalformedModelException {
    var bytes = new ByteArrayOutputStream();
    try (var out = new DataOutputStream(bytes)) {
      parameter.save(out);
    }
    try (var previous = parameter.getArray();
        var in = new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      parameter.load(manager, in);
      var fresh = (PtNDArray) parameter.getArray();
      fresh.setRequiresGradient(true);
      Assert.assertNull(JniUtils.getGradient(fresh));
    }
  }

  /** 自動微分が呼んだ演算を実行し、逆伝播で使う中間値と書き出した勾配を記録する。 */
  private static TrainingKernels recording(
      TrainingKernels kernels, TrainingTensorTrace trace, Set<String> called, Calls calls) {
    var candidates = kernels.candidateKernels();
    Assert.assertNotNull(candidates, "missing candidate training kernels");
    Assert.assertTrue(kernels instanceof RelationCodes, "missing relation code kernel");
    return (TrainingKernels)
        Proxy.newProxyInstance(
            TrainingStepRecorder.class.getClassLoader(),
            new Class<?>[] {TrainingKernels.class, RelationCodes.class, CandidateKernels.class},
            (self, method, args) -> {
              String op = method.getName();
              if (op.equals("candidateKernels")) return self;
              try {
                method.invoke(
                    method.getDeclaringClass() == CandidateKernels.class ? candidates : kernels,
                    args);
              } catch (InvocationTargetException e) {
                throw e.getCause();
              }
              int first, last = args.length;
              switch (op) {
                case "codes", "count" -> first = last - 1;
                case "countGradient" -> first = last - 3;
                case "candidateAttention", "candidateAttentionGradient" -> first = 1;
                case "forward", "backward" -> {
                  boolean forward = op.equals("forward");
                  if (args[0] instanceof RelationAttention.Dimensions d) {
                    first = last - (forward ? 3 : 4);
                    op = (d.causalFixedTokens() > 0 ? "causal/" : "relation/") + op;
                  } else {
                    first = last - (forward ? 2 : 3);
                    op = "norm/" + op;
                  }
                }
                default -> throw new AssertionError("unrecognized training operation " + op);
              }
              called.add(op);
              if (calls != null) calls.save(op, args);
              if (trace != null)
                for (int i = first; i < last; i++) trace.tensor(op + "/" + i, (NDArray) args[i]);
              return null;
            });
  }
}
