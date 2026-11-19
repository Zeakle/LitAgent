// Refresh full node payloads separately from the lightweight SSE timeline.
export function createLiveViewRefresh({ current, load, apply, onError, delay = 1000 }) {
  let timer = null;
  let generation = 0;
  let busy = false;
  let dirty = false;
  function cancel() {
    generation += 1;
    clearTimeout(timer); timer = null; dirty = false; busy = false;
  }
  function schedule() {
    dirty = true;
    if (timer || busy) return;
    const state = current();
    if (!state.view?.identity) return;
    const { identity } = state.view;
    const epoch = state.requestEpoch;
    const captured = generation;
    timer = setTimeout(async () => {
      timer = null; busy = true; dirty = false;
      try {
        const view = await load(identity);
        const now = current();
        if (generation === captured && now.requestEpoch === epoch
          && now.view?.identity?.key === identity.key
          && !now.view?.displayRevision) apply(view);
      } catch (error) {
        if (generation === captured) onError?.(error);
      } finally {
        if (generation === captured) { busy = false; if (dirty) schedule(); }
      }
    }, delay);
  }
  return { schedule, cancel };
}
