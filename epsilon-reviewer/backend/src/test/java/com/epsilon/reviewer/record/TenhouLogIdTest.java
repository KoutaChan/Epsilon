package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.net.URI;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 天鳳の牌譜の URL の規則（主・問い・正規表現）。雀魂の URL はネットワークの前に断る。 */
public class TenhouLogIdTest {
  private static final String ID = "2026091512gm-0089-0000-1a2b3c4d";

  @DataProvider
  public Object[][] mahjongSoulHosts() {
    return new Object[][] {
      {"game.mahjongsoul.com"},
      {"mahjongsoul.game.yo-star.com"},
      {"game.maj-soul.com"},
      {"game.maj-soul.net"}
    };
  }

  @Test(dataProvider = "mahjongSoulHosts")
  public void mahjongSoulUrlsAreUnsupported(String host) {
    for (String scheme : List.of("http", "https"))
      refused(
          scheme + "://" + host + "/?paipu=260915-12345678-1234-1234-1234-123456789abc_a123",
          ErrorCode.UNSUPPORTED_URL);
  }

  @Test
  public void tenhouUrlsGiveTheLogId() {
    for (String url :
        List.of(
            "https://tenhou.net/0/?log=" + ID + "&tw=1",
            "http://tenhou.net/0/?log=" + ID,
            "https://www.tenhou.net/3/?log=" + ID,
            "https://TENHOU.net/0/?log=" + ID,
            "https://tenhou.net/0/log/?" + ID,
            "https://tenhou.net/0/?log=other&log=" + ID))
      Assert.assertEquals(TenhouLogId.parse(url).value(), ID, url);
    Assert.assertEquals(
        TenhouLogId.parse("https://tenhou.net/0/?log=" + ID).downloadUri(),
        URI.create("https://tenhou.net/0/log/?" + ID));
  }

  @Test
  public void otherShapesAreInvalid() {
    for (String url :
        List.of(
            "http://[bad",
            "tenhou.net/0/?log=" + ID,
            "ftp://tenhou.net/0/?log=" + ID,
            "https://tenhou.net:8443/0/?log=" + ID,
            "https://user@tenhou.net/0/?log=" + ID,
            "https://tenhou.net/0/?log=2026091512gm-0089-0000-1a2b3c4",
            "https://tenhou.net/0/?log=%ZZ",
            "https://tenhou.net/0/"))
      refused(url, ErrorCode.INVALID_URL);
  }

  private static void refused(String url, ErrorCode code) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> TenhouLogId.parse(url));
    Assert.assertEquals(failure.code(), code, url);
  }
}
