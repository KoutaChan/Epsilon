import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("epsilonReviewer", {
  request: (request: unknown) =>
    ipcRenderer.invoke("reviewer:request", request),
  getSettings: () => ipcRenderer.invoke("reviewer:settings"),
  setServerUrl: (url: string) => ipcRenderer.invoke("reviewer:server", url),
  setLanguage: (language: string) =>
    ipcRenderer.invoke("reviewer:language", language),
  chooseResultsDirectory: () => ipcRenderer.invoke("reviewer:directory"),
});
