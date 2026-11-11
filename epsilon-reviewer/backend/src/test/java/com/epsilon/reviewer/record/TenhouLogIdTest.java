package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.net.URI;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 天鳳の牌譜の URL の形（ホスト・クエリ・id の正規表現）を確かめる。雀魂の URL は、ネットワークにつなぐ前に断る。 */
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

  /** ロビーの番号は 5 桁のこともある。 */
  @Test
  public void fiveDigitLobbiesAreRead() {
    String id = "2012060420gm-0009-10011-acfd4b57";
    Assert.assertEquals(TenhouLogId.parse("https://tenhou.net/0/?log=" + id).value(), id);
    Assert.assertEquals(TenhouLogId.parse("https://tenhou.net/0/log/?" + id).value(), id);
  }

  /**
   * 自分の対局の牌譜を開いた URL の id は、末尾が x と 16 進 12 桁。牌譜を取るときの id（末尾が 16 進 8 桁）に直して読む。直し方は対局の
   * 日時で変わり、2010041111 までの id は表の先頭を使う。
   */
  @Test
  public void ownGameIdsBecomeTheDownloadId() {
    Object[][] pairs = {
      {"2011033107gm-0089-0000-xdee083a035b1", "2011033107gm-0089-0000-8afe0a3f"},
      {"2014022323gm-0009-0000-x2f8a1ef6c2d0", "2014022323gm-0009-0000-95049099"},
      {"2026091512gm-0089-0000-x0123456789ab", "2026091512gm-0089-0000-60649837"},
      {"2010041111gm-0009-0000-x2f8a1ef6c2d0", "2010041111gm-0009-0000-670447b1"},
      {"2010010100gm-0009-0000-xffff00001234", "2010010100gm-0009-0000-a98789a3"}
    };
    for (Object[] pair : pairs) {
      TenhouLogId id = TenhouLogId.parse("http://tenhou.net/0/?log=" + pair[0] + "&tw=0");
      Assert.assertEquals(id.value(), pair[1]);
      Assert.assertEquals(id.downloadUri(), URI.create("https://tenhou.net/0/log/?" + pair[1]));
    }
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
            "https://tenhou.net/0/?log=2026091512gm-0089-000-1a2b3c4d",
            "https://tenhou.net/0/?log=2026091512gm-0089-100000-1a2b3c4d",
            "https://tenhou.net/0/?log=2026091512gm-0089-0000-x0123456789a",
            "https://tenhou.net/0/?log=2026091512gm-0089-0000-x0123456789abc",
            "https://tenhou.net/0/?log=2026091512gm-0089-0000-x0123456789ag",
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
