export function createWorkbenchStore(initial = {}) {
  const defaultExecutionUi = () => ({
    mode: "overview",
    expandedStageIds: [],
    selectedNodeId: null,
    scale: 1,
    translateX: 0,
    translateY: 0,
    timelineFilter: "all",
    historyCollapsed: false,
    fittedSignature: null,
  });
  let state = {
    locale: "zh-CN",
    runs: [],
    selectedRunKey: null,
    activeSection: "report",
    sectionWasUserSelected: false,
    view: null,
    inspector: null,
    loading: false,
    error: null,
    requestEpoch: 0,
    eventCursor: 0,
    draftReviewState: {},
    executionUiByRun: {},
    ...initial,
  };
  const listeners = new Set();

  function notify() {
    for (const listener of listeners) listener(state);
  }

  return {
    get: () => state,
    update(patch) {
      state = { ...state, ...patch };
      notify();
    },
    mutate(mutator) {
      const next = mutator({ ...state });
      if (next) state = next;
      notify();
    },
    beginRequest() {
      const requestEpoch = state.requestEpoch + 1;
      state = { ...state, requestEpoch, loading: true, error: null };
      notify();
      return requestEpoch;
    },
    isCurrent(epoch) {
      return state.requestEpoch === epoch;
    },
    getExecutionUi(runId) {
      return { ...defaultExecutionUi(), ...(state.executionUiByRun[runId] || {}) };
    },
    setExecutionUi(runId, patch) {
      if (!runId) return;
      const current = { ...defaultExecutionUi(), ...(state.executionUiByRun[runId] || {}) };
      state = {
        ...state,
        executionUiByRun: {
          ...state.executionUiByRun,
          [runId]: { ...current, ...patch },
        },
      };
      notify();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
