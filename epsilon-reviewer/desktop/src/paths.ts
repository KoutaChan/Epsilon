import path from "node:path";

/** アプリが読み書きする場所。 */
export interface AppPaths {
  userData: string;
  /** web/dist（epsilon-reviewer://app の中身）。 */
  web: string;
  /** language/<id>/。 */
  language: string;
  /** Java のエンジン（runtime・lib・native・models）。 */
  engine: string;
  icon: string;
  preload: string;
}

/**
 * 梱包では resources の下、未梱包では epsilon-reviewer の作業の木（../web/dist・../language・desktop/engine）。
 *
 * @param appPath app.getAppPath()（未梱包は desktop、梱包は resources/app.asar）
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
