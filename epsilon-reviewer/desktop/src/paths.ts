import path from "node:path";

/** アプリが読み書きするフォルダーとファイル。 */
export interface AppPaths {
  userData: string;
  /** web/dist。epsilon-reviewer://app で返す画面のファイルが入る。 */
  web: string;
  /** 言語ごとの language/<id>/ を並べたフォルダー。 */
  language: string;
  /** Java のエンジンのフォルダー。runtime・lib・native・models が入る。 */
  engine: string;
  icon: string;
  preload: string;
}

/**
 * アプリが使う場所を返す。パッケージしたアプリでは resources の下を使う。パッケージしていないときは
 * epsilon-reviewer の作業ツリー（../web/dist・../language・desktop/engine）を使う。
 *
 * @param appPath app.getAppPath() の値（パッケージしていなければ desktop、していれば resources/app.asar）
 */
export function appPaths(
  packaged: boolean,
  resourcesPath: string,
  appPath: string,
  userData: string,
): AppPaths {
  const preload = path.join(appPath, "dist", "preload.cjs");
  if (packaged)
    return {
      userData,
      web: path.join(resourcesPath, "web"),
      language: path.join(resourcesPath, "language"),
      engine: path.join(resourcesPath, "engine"),
      icon: path.join(resourcesPath, "icon.png"),
      preload,
    };
  const reviewer = path.dirname(appPath);
  return {
    userData,
    web: path.join(reviewer, "web", "dist"),
    language: path.join(reviewer, "language"),
    engine: path.join(appPath, "engine"),
    icon: path.join(reviewer, "assets", "icon.png"),
    preload,
  };
}
