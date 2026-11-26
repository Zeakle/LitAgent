/* Measure the production form and rendered workbench using one browser clock. */
async ({query, timeoutMs}) => {
  const get = selector => document.querySelector(selector);
  const newRun = get("#open-new-run");
  if (!newRun || newRun.disabled) throw new Error("workbench_submit_unavailable");
  const prior = new URL(location.href).searchParams.get("run");
  newRun.click();
  get("#query-input").value = query;
  get("#report-language").value = "en";
  const jev = get("#verification-jev");
  jev.checked = false;
  jev.dispatchEvent(new Event("change"));
  const markers = {};
  const started = performance.now();
  const mark = name => { markers[name] ??= Math.round(performance.now() - started); };
  const terminalStates = new Set(["completed", "failed", "cancelled", "rejected", "interrupted"]);

  return new Promise(resolve => {
    let status, runId, graphOpened = false, draftOpened = false, reportOpened = false;
    let frame = null, settled = false;
    const finish = error => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(reportGrace);
      if (frame !== null) cancelAnimationFrame(frame);
      observer.disconnect();
      resolve({runId, status, markers, error});
    };
    let reportGrace;
    const scan = () => {
      frame = null;
      const current = new URL(location.href).searchParams.get("run");
      if (!current || current === prior) {
        const notice = get("#survey-submit-status");
        if (!notice?.hidden && notice?.textContent.trim()
            && !get('#new-run-form button[type="submit"]')?.disabled) {
          finish("survey_submission_unconfirmed");
        }
        return;
      }
      const [source, currentId] = current.split(":");
      if (!["legacy", "archive"].includes(source) || (runId && runId !== currentId)) return;
      runId = currentId;
      const progress = get("#execution-progress");
      if (progress?.dataset.runId !== runId || progress.hidden) return;
      const state = progress.dataset.status;
      if (state === "running" && get("#execution-current-steps button")) mark("first_progress");
      if (!graphOpened && !reportOpened && state) {
        graphOpened = true;
        get('[data-execution-mode="dag"]')?.click();
      }
      if (!draftOpened) {
        const node = get('#execution-graph [data-node-id$=":synthesis:r1"][data-status="completed"]');
        if (node && node.getClientRects().length) {
          draftOpened = true;
          node.click();
        }
      }
      const draft = get("#inspector-content .execution-document");
      if (draftOpened && draft?.textContent.trim() && draft.getClientRects().length) {
        mark("first_draft");
      }
      if (!terminalStates.has(state)) return;
      status = state;
      mark("terminal");
      if (status !== "completed") return finish(null);
      if (!reportOpened) {
        reportOpened = true;
        get("#close-inspector")?.click();
        get('[data-section="report"]')?.click();
        reportGrace = setTimeout(() => finish("final_report_unavailable"), 5000);
      }
      const report = get("#report-document");
      if (report?.getClientRects().length && report.textContent.trim() &&
          (report.querySelector("h1, h2, h3") || report.classList.contains("render-fallback"))) {
        mark("final_report");
        finish(null);
      }
    };
    const observer = new MutationObserver(() => {
      if (frame === null) frame = requestAnimationFrame(scan);
    });
    observer.observe(document.body, {
      childList: true, subtree: true, characterData: true, attributes: true,
      attributeFilter: ["data-status", "data-run-id", "hidden", "aria-hidden"],
    });
    const timer = setTimeout(() => finish("workbench_timeout"), timeoutMs);
    get("#new-run-form").requestSubmit();
  });
}
