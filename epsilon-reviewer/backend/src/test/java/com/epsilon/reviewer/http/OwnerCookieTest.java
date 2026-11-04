package com.epsilon.reviewer.http;

import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.store.OwnerId;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 持ち主の Cookie の読みと発行、持ち主の id、owner の道だけが Cookie を扱うこと。 */
public class OwnerCookieTest {
  private static final String VALID = "A".repeat(43);

  @Test
  public void theLastValidKeyOfAllCookieHeadersWins() {
    String name = OwnerCookie.NAME;
    Assert.assertEquals(OwnerCookie.key(List.of("a=b; " + name + "=" + VALID + "; c=d")), VALID);
    Assert.assertEquals(
        OwnerCookie.key(List.of(name + "=" + VALID, name + "=" + "B".repeat(43))), "B".repeat(43));
    Assert.assertEquals(
        OwnerCookie.key(List.of(name + "=" + VALID + "; " + name + "=" + "C".repeat(42))), VALID);
    Assert.assertNull(OwnerCookie.key(List.of(name + "=" + "!".repeat(43))));
    Assert.assertNull(OwnerCookie.key(List.of()));
  }

  @Test
  public void ownerRoutesIssueTheCookieOnceAndPublicRoutesNever() {
    try (var server = TestServer.start()) {
      var history = server.send(server.request("/api/history", null));
      Assert.assertEquals(history.statusCode(), 200);
      String cookie = history.headers().firstValue("Set-Cookie").orElseThrow();
      Assert.assertTrue(
          cookie.matches(
              OwnerCookie.NAME
                  + "=[A-Za-z0-9_-]{43}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000"),
          cookie);
      String pair = cookie.split(";", 2)[0];
      var again =
          server.send(server.request("/api/history", null).header("Cookie", "x=1; " + pair));
      Assert.assertFalse(again.headers().firstValue("Set-Cookie").isPresent());
      var malformed =
          server.send(
              server
                  .request("/api/history", null)
                  .header("Cookie", OwnerCookie.NAME + "=short"));
      Assert.assertTrue(malformed.headers().firstValue("Set-Cookie").isPresent());
      var missing = server.send(server.request("/api/jobs/nope", null));
      Assert.assertTrue(missing.headers().firstValue("Set-Cookie").isPresent());
      for (String path : new String[] {"/api/models", "/api/shares/nope", "/api/nothing", "/"})
        Assert.assertFalse(
            server.send(server.request(path, null)).headers().firstValue("Set-Cookie").isPresent(),
            path);
    }
  }

  @Test
  public void httpsOriginsMakeTheCookieSecure() {
    try (var server = TestServer.start(URI.create("https://127.0.0.1:0"), null)) {
      var history = server.send(server.request("/api/history", null));
      Assert.assertEquals(history.statusCode(), 200, history.body());
      Assert.assertTrue(
          history.headers().firstValue("Set-Cookie").orElseThrow().endsWith("; Secure"));
    }
  }

  @Test
  public void theOwnerIdIsTheSha256OfTheKey() throws Exception {
    String expected =
        HexFormat.of()
            .formatHex(
                MessageDigest.getInstance("SHA-256")
                    .digest(VALID.getBytes(StandardCharsets.US_ASCII)));
    Assert.assertEquals(OwnerId.fromCookieKey(VALID).value(), expected);
    Assert.assertEquals(OwnerId.fromCookieKey(VALID), TestReviewer.OWNER);
    Assert.assertEquals(OwnerId.LOCAL.value(), "0".repeat(64));
  }
}
