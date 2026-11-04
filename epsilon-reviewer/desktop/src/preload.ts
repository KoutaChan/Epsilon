import { contextBridge, ipcRenderer, webUtils } from "electron";
import { BRIDGE_CHANNELS, type ReviewerBridge } from "../../protocol/bridge.ts";

const bridge: ReviewerBridge = {
  invoke: (command, request, body) =>
    body === undefined
      ? ipcRenderer.invoke(BRIDGE_CHANNELS.invoke, command, request)
      : ipcRenderer.invoke(BRIDGE_CHANNELS.invoke, command, request, body),
  pathForFile: (file) => webUtils.getPathForFile(file),
  settings: () => ipcRenderer.invoke(BRIDGE_CHANNELS.settings),
  setLanguage: (language) =>
    ipcRenderer.invoke(BRIDGE_CHANNELS.language, language),
  chooseResultsDirectory: () => ipcRenderer.invoke(BRIDGE_CHANNELS.directory),
  exportResult: (resultId) =>
    ipcRenderer.invoke(BRIDGE_CHANNELS.export, resultId),
};

contextBridge.exposeInMainWorld("epsilonReviewer", bridge);
