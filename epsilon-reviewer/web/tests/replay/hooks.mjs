// フックを本物の React（react-dom/client）で動かす。ブラウザーが無いので、何も描かない部品を、React が触る所だけを
// 持つ偽の要素に載せる。状態の更新と effect は、ブラウザーの中と同じ順に進む。

const listeners = { addEventListener() {}, removeEventListener() {} };
globalThis.window = globalThis;
globalThis.HTMLIFrameElement = class {};
globalThis.document = { ...listeners, nodeType: 9 };
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
// react-dom は読み込むときに window を見るので、上の用意の後で読み込む。
const { act, createElement } = await import("react");
const { createRoot } = await import("react-dom/client");

/**
 * useHook(props) を呼ぶ部品を載せる。seen には、フックの返した値のうち子の部品へ渡ったものが描画の順に入る
 * （描画の途中で状態を直して描き直した分の、捨てられた値は入らない）。
 */
export async function mountHook(useHook, props) {
  const seen = [];
  const Child = ({ value }) => {
    seen.push(value);
    return null;
  };
  const Host = ({ value }) => createElement(Child, { value: useHook(value) });
  const root = createRoot({
    ...listeners,
    nodeType: 1,
    tagName: "DIV",
    ownerDocument: document,
  });
  const render = (value) =>
    act(() => root.render(createElement(Host, { value })));
  await render(props);
  return {
    seen,
    /** いちばん新しい描画でフックが返した値。 */
    get current() {
      return seen.at(-1);
    },
    /** 別の props で描き直す。 */
    render,
    /** 状態を変える操作を行い、その描画と effect が済むまで待つ。 */
    act,
    unmount: () => act(() => root.unmount()),
  };
}
