package com.epsilon.reviewer.api;

import com.epsilon.reviewer.error.ErrorCode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * protocol/commands.ts の COMMANDS と protocol/errors.ts の SERVER_ERROR_CODES を読み、Java の Command と
 * ErrorCode と、並びの順も中身も一致することを確かめる。
 */
public class ProtocolContractTest {
  private static final Path PROTOCOL = Path.of("..", "protocol");
  private static final Pattern BLOCK = Pattern.compile("^  (\\w+): \\{$");
  private static final Pattern HTTP =
      Pattern.compile("^    http: \\{ method: \"(\\w+)\", path: \"([^\"]+)\" },$");
  private static final Pattern STDIO = Pattern.compile("^    stdio: (true|false),$");
  private static final Pattern RENDERER = Pattern.compile("^    renderer: (true|false),$");
  private static final Pattern ACCESS = Pattern.compile("^    access: \\[(.*)],$");
  private static final Pattern REQUEST = Pattern.compile("^    request: \"(\\w+)\",$");
  private static final Pattern RESPONSE = Pattern.compile("^    response: \"(\\w+)\",$");
  private static final Pattern CODE = Pattern.compile("^  \"(\\w+)\",$");

  @Test
  public void commandsMatchTheProtocolTable() throws IOException {
    List<Spec> specs = commands();
    Assert.assertEquals(
        specs.stream().map(Spec::name).toList(),
        Arrays.stream(Command.values()).map(Command::wireName).toList());
    for (Command command : Command.values()) {
      Spec spec = specs.get(command.ordinal());
      String where = command.wireName();
      Assert.assertEquals(command.httpMethod(), spec.method(), where);
      Assert.assertEquals(command.httpPath(), spec.path(), where);
      Assert.assertEquals(command.stdio(), spec.stdio(), where);
      Assert.assertEquals(
          command.access().stream().map(access -> lower(access.name())).toList(),
          spec.access(),
          where);
      Assert.assertEquals(lower(command.requestBody().name()), spec.request(), where);
      Assert.assertEquals(lower(command.responseBody().name()), spec.response(), where);
    }
  }

  @Test
  public void errorCodesMatchTheServerErrorCodes() throws IOException {
    List<String> lines = Files.readAllLines(PROTOCOL.resolve("errors.ts"));
    int start = lines.indexOf("export const SERVER_ERROR_CODES = [");
    List<String> codes = new ArrayList<>();
    for (int i = start + 1; !lines.get(i).equals("] as const;"); i++) {
      Matcher code = CODE.matcher(lines.get(i));
      Assert.assertTrue(code.matches(), lines.get(i));
      codes.add(code.group(1));
    }
    Assert.assertEquals(
        codes, Arrays.stream(ErrorCode.values()).map(ErrorCode::code).toList());
  }

  /** COMMANDS の表を読む。1 つのコマンドが 1 つのブロックで、ブロックの各行は決まった書き方に従う。知らない行があれば失敗にする。 */
  private static List<Spec> commands() throws IOException {
    List<String> lines = Files.readAllLines(PROTOCOL.resolve("commands.ts"));
    int start = lines.indexOf("export const COMMANDS = {");
    Assert.assertTrue(start >= 0, "COMMANDS is missing.");
    List<Spec> specs = new ArrayList<>();
    for (int i = start + 1; !lines.get(i).startsWith("} as const"); i++) {
      Matcher block = BLOCK.matcher(lines.get(i));
      Assert.assertTrue(block.matches(), lines.get(i));
      String method = null;
      String path = null;
      Boolean stdio = null;
      List<String> access = null;
      String request = null;
      String response = null;
      for (i++; !lines.get(i).equals("  },"); i++) {
        String line = lines.get(i);
        Matcher matcher;
        if ((matcher = HTTP.matcher(line)).matches()) {
          method = matcher.group(1);
          path = matcher.group(2);
        } else if ((matcher = STDIO.matcher(line)).matches())
          stdio = Boolean.parseBoolean(matcher.group(1));
        else if ((matcher = ACCESS.matcher(line)).matches())
          access =
              Arrays.stream(matcher.group(1).split(","))
                  .map(value -> value.trim().replace("\"", ""))
                  .toList();
        else if ((matcher = REQUEST.matcher(line)).matches()) request = matcher.group(1);
        else if ((matcher = RESPONSE.matcher(line)).matches()) response = matcher.group(1);
        else if (!line.equals("    http: null,") && !RENDERER.matcher(line).matches())
          Assert.fail("Unexpected line in COMMANDS: " + line);
      }
      specs.add(new Spec(block.group(1), method, path, stdio, access, request, response));
    }
    return specs;
  }

  private static String lower(String name) {
    return name.toLowerCase(Locale.ROOT);
  }

  private record Spec(
      String name,
      String method,
      String path,
      Boolean stdio,
      List<String> access,
      String request,
      String response) {}
}
