/* Only the current selection may update the read-only benchmark workspace. */
export function createBenchmarkController({ api, state, render, renderObservation, clearObservation, onError }) {
  let loadSequence = 0, selectionSequence = 0, observationSequence = 0;
  async function load() {
    const sequence = ++loadSequence;
    const selectionAtStart = selectionSequence;
    try {
      const results = await api.json("/workbench/benchmarks");
      if (sequence !== loadSequence) return;
      state.results = results;
      state.loaded = true;
      const preferred = state.selectedId || results[0]?.benchmark_id;
      // A manual selection made while the index loaded owns the detail panel.
      if (preferred && selectionAtStart === selectionSequence) await select(preferred);
      else render();
    } catch (error) {
      if (sequence === loadSequence && error?.name !== "AbortError") onError(error);
    }
  }

  async function select(id) {
    const sequence = ++selectionSequence;
    ++observationSequence;
    state.selectedId = id;
    state.detail = null;
    state.observation = null;
    render();
    try {
      const detail = await api.json(`/workbench/benchmarks/${encodeURIComponent(id)}`);
      if (sequence !== selectionSequence || state.selectedId !== id) return;
      state.detail = detail;
      render();
    } catch (error) {
      if (sequence === selectionSequence && error?.name !== "AbortError") onError(error);
    }
  }

  async function selectObservation(id) {
    const benchmarkId = state.selectedId;
    const selection = selectionSequence;
    const sequence = ++observationSequence;
    state.observation = null;
    clearObservation();
    try {
      const payload = await api.json(`/workbench/benchmarks/${encodeURIComponent(benchmarkId)}/observations/${encodeURIComponent(id)}`);
      if (sequence !== observationSequence || selection !== selectionSequence || benchmarkId !== state.selectedId) return;
      state.observation = payload;
      renderObservation(payload);
    } catch (error) {
      if (sequence === observationSequence && selection === selectionSequence && error?.name !== "AbortError") onError(error);
    }
  }

  return { load, select, selectObservation };
}
