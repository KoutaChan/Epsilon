package com.epsilon.agent.run;

import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/** control/done に移した要求（従った要求と取り消した要求）を読むテスト用のヘルパー。 */
final class RetiredRequests {

  private RetiredRequests() {}

  /** @return control/done にある要求（番号の順）。control/done が無ければ空 */
  static List<ControlRequest> read(RunDirectory run) throws IOException {
    Path done = run.control().resolve("done");
    if (!Files.isDirectory(done)) {
      return List.of();
    }
    List<ControlRequest> requests = new ArrayList<>();
    try (Stream<Path> files = Files.list(done)) {
      for (Path file : files.filter(path -> path.getFileName().toString().endsWith(".json")).toList()) {
        requests.add(RecordFile.read(ControlRequest.class, file).value());
      }
    }
    requests.sort(Comparator.comparingInt(ControlRequest::number));
    return requests;
  }
}
