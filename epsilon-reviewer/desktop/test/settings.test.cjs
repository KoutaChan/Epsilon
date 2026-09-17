const { test } = require("node:test");
const assert = require("node:assert/strict");
const { mkdtemp, rm } = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { SettingsStore } = require("../dist/settings.js");
const { Languages } = require("../dist/language.js");

test("language and server changes commit in order and survive restart", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "review-settings-"));
  try {
    const file = path.join(directory, "settings.json");
    const languages = new Languages(path.resolve(__dirname, "../../language"));
    const defaults = {
      language: "ja",
      serverUrl: "http://127.0.0.1:8080",
      resultsDirectory: directory,
    };
    const settings = await SettingsStore.open(file, defaults, languages);
    await Promise.all([
      settings.setLanguage("en"),
      settings.setServerUrl("https://review.example.com"),
    ]);
    const reopened = await SettingsStore.open(file, defaults, languages);
    assert.equal(reopened.value.language, "en");
    assert.equal(reopened.value.serverUrl, "https://review.example.com");
    assert.equal(
      languages.getMessage(reopened.value.language, "desktop.exportTitle"),
      "Save analysis result",
    );
    assert.throws(
      () => reopened.setServerUrl("https://user:password@example.com"),
      (error) => error.code === "invalid_server_url",
    );
    assert.throws(
      () => reopened.setLanguage("../unknown"),
      (error) => error.code === "invalid_language",
    );
    assert.equal(
      (await SettingsStore.open(file, defaults, languages)).value.language,
      "en",
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
