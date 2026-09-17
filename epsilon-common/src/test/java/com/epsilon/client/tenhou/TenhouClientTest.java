package com.epsilon.client.tenhou;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 系列ごとの既定値と既存 CLI の長短オプションを、接続せずに検証する。 */
public class TenhouClientTest {

  @DataProvider
  public Object[][] optionNames() {
    return new Object[][] {{"--checkpoint", "--room", "--name"}, {"-c", "-r", "-n"}};
  }

  @Test(dataProvider = "optionNames")
  public void longAndShortOptionsKeepWindowsPathsSeparateFromRoomAndPolicy(
      String checkpointFlag, String roomFlag, String nameFlag) {
    var options =
        TenhouClient.Options.parse(
            new String[] {
              checkpointFlag,
              "C:\\models\\best",
              roomFlag,
              "C12345678",
              nameFlag,
              "Epsilon テスト",
              "--uri",
              "ws://localhost:1234",
              "--sample"
            },
            "checkpoints/test");
    Assert.assertEquals(
        options,
        new TenhouClient.Options(
            "C:\\models\\best", "C12345678", "Epsilon テスト", "ws://localhost:1234", true));
  }

  @Test
  public void independentParsesUseTheCallingSeriesDefaultWithoutRetainingPriorOptions() {
    var first =
        TenhouClient.Options.parse(new String[] {"--room", "C1", "--sample"}, "checkpoints/first");
    var second = TenhouClient.Options.parse(new String[] {"--room", "C2"}, "checkpoints/second");
    Assert.assertEquals(first.checkpointDir(), "checkpoints/first");
    Assert.assertTrue(first.samplePolicy());
    Assert.assertEquals(second.checkpointDir(), "checkpoints/second");
    Assert.assertFalse(second.samplePolicy());
  }

  @DataProvider
  public Object[][] usageRequests() {
    return new Object[][] {
      {new String[] {}},
      {new String[] {"--room"}},
      {new String[] {"--help"}},
      {new String[] {"--room", "C1", "-h"}}
    };
  }

  @Test(dataProvider = "usageRequests")
  public void helpAndMissingRoomReturnWithoutStartingASession(String[] arguments) {
    Assert.assertNull(TenhouClient.Options.parse(arguments, "checkpoints/test"));
  }
}
