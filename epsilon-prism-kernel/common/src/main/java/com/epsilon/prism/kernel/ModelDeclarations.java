package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.ModelSchema.Element;
import com.epsilon.prism.kernel.ModelSchema.FeedForward;
import com.epsilon.prism.kernel.ModelSchema.Input;
import com.epsilon.prism.kernel.ModelSchema.Layer;
import com.epsilon.prism.kernel.ModelSchema.Mask;
import com.epsilon.prism.kernel.ModelSchema.Model;
import com.epsilon.prism.kernel.ModelSchema.Queries;
import com.epsilon.prism.kernel.ModelSchema.Role;
import com.epsilon.prism.kernel.ModelSchema.Section;
import com.epsilon.prism.kernel.ModelSchema.Segment;
import com.epsilon.prism.kernel.ModelSchema.Source;
import com.epsilon.prism.kernel.ModelSchema.Stage;
import com.epsilon.prism.kernel.ModelSchema.Weight;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.List;
import java.util.function.Function;

/** モデル宣言に ABI バージョンと長さを付け、Java と native の間で受け渡す。 */
public final class ModelDeclarations {
  private ModelDeclarations() {}

  public static ByteBuffer encode(Model model) {
    validate(model);
    var body = new DeclarationIO.Output();
    model.write(body);
    var header = new DeclarationIO.Output();
    header.word(ModelSchema.MAGIC);
    header.positive(ModelSchema.VERSION);
    header.size(body.bytes.size());
    int total = Math.addExact(header.bytes.size(), body.bytes.size());
    return ByteBuffer.allocateDirect(total)
        .put(header.bytes.toByteArray())
        .put(body.bytes.toByteArray())
        .flip()
        .asReadOnlyBuffer()
        .order(ByteOrder.LITTLE_ENDIAN);
  }

  public static Model decode(ByteBuffer bytes, int length, int abi) {
    if (length != bytes.remaining() || abi != ModelSchema.VERSION)
      throw new IllegalArgumentException("Declaration ABI or byte length mismatch");
    var in = new DeclarationIO.Input(bytes);
    if (in.word() != ModelSchema.MAGIC)
      throw new IllegalArgumentException("Invalid declaration magic");
    int version = in.positive();
    if (version != abi)
      throw new IllegalArgumentException(
          "Unsupported declaration ABI: " + version + ", expected " + abi);
    if (in.size() != in.buffer.remaining())
      throw new IllegalArgumentException("Declaration byte length mismatch");
    Model model = Model.read(in);
    if (in.buffer.hasRemaining())
      throw new IllegalArgumentException("Unknown required declaration fields");
    validate(model);
    return model;
  }

  public static void validate(Model model) {
    int fixed = Math.addExact(1, Math.addExact(model.seats(), model.tiles()));
    Math.addExact(fixed, model.maxEvents());
    Math.multiplyExact(model.heads(), model.headWidth());
    if (model.layers().isEmpty()
        || model.inputs().isEmpty()
        || model.stages().isEmpty()
        || model.roles().isEmpty()
        || model.weights().isEmpty()
        || model.arithmetic().isEmpty())
      throw new IllegalArgumentException("Missing model components");
    unique(model.inputs(), Input::name);
    unique(model.weights(), Weight::name);
    unique(model.roles(), Role::name);
    for (Input input : model.inputs()) {
      Section section = Section.valueOf(input.name());
      if (section.axis() != input.axis() || section.element() != input.element())
        throw new IllegalArgumentException("Invalid input axes or element: " + input.name());
      if (!input.vocabularies().isEmpty()
          && (input.element() != Element.U8 || input.width() != input.vocabularies().size()))
        throw new IllegalArgumentException("Invalid column vocabulary for " + input.name());
      for (int n : input.vocabularies())
        if (n < 1 || n > 256) throw new IllegalArgumentException("Vocabulary outside UINT8");
    }
    for (int i = 0; i < model.layers().size(); i++) {
      Layer layer = model.layers().get(i);
      if (layer.source() != (i == 0 ? Source.EMBEDDING : Source.PREVIOUS_LAYER))
        throw new IllegalArgumentException("Invalid layer input at " + i);
      if (layer.ffn() == FeedForward.SHARED && layer.contextWidth() != layer.tileWidth())
        throw new IllegalArgumentException("Shared FFN widths differ");
      if (layer.mask() == Mask.PUBLIC_PREFIX && layer.queries() != Queries.ALL_TOKENS)
        throw new IllegalArgumentException("Public prefix needs event queries");
    }
    int column = fixed, row = 0;
    for (Segment segment : model.segments()) {
      if (segment.column() != column || segment.firstRow() != row || segment.rows() > 256)
        throw new IllegalArgumentException("Invalid candidate segment " + segment.parameter());
      column = Math.addExact(column, model.tiles());
      row = Math.addExact(row, segment.rows());
    }
    if (column != model.codeWidth()
        || row != model.tableRows()
        || model.stages().contains(Stage.SEGMENT_BIAS) != !model.segments().isEmpty()
        || model.stages().contains(Stage.COUNT_TILES) != (model.countWidth() > 0))
      throw new IllegalArgumentException("Candidate axes do not match stages");
    weightBytes(model);
  }

  public static long weightBytes(Model model) {
    long total = 0;
    for (Weight weight : model.weights()) {
      if (weight.element() != Element.FP32)
        throw new IllegalArgumentException("Master weights must be FP32");
      long bytes = Float.BYTES;
      for (int axis : weight.axes()) {
        DeclarationIO.positive(axis);
        bytes = Math.multiplyExact(bytes, axis);
      }
      total = Math.addExact(total, bytes);
    }
    return total;
  }

  private static <T> void unique(List<T> rows, Function<T, String> name) {
    var names = new HashSet<String>();
    for (T row : rows)
      if (!names.add(name.apply(row)))
        throw new IllegalArgumentException("Duplicate model entry: " + name.apply(row));
  }
}
