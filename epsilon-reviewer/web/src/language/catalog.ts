import japanese from "../../../language/ja.json";

export type MessageKey = keyof typeof japanese;
export type Messages = Record<MessageKey, string>;
export type Translate = (
  key: MessageKey,
  values?: Record<string, string | number>,
) => string;
const modules = import.meta.glob<Messages>("../../../language/*.json", {
  eager: true,
  import: "default",
});
export const languages = Object.entries(modules).map(([path, messages]) => ({
  id: path.slice(path.lastIndexOf("/") + 1, -5),
  name: messages["language.name"],
  messages,
}));
export function translate(language: string): Translate {
  const messages = languages.find((entry) => entry.id === language)!.messages;
  return (key, values) =>
    messages[key].replace(/\{(\w+)\}/g, (token, name: string) =>
      String(values?.[name] ?? token),
    );
}
export function errorKey(error: unknown): MessageKey {
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? String(error.code)
      : "operation_failed";
  const key = `errors.${code}`;
  return key in japanese ? (key as MessageKey) : "errors.operation_failed";
}
