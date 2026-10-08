/* Deterministic format requests share the selected revision and request epoch. */
export function bindFormatControls({ store, api, t, setNotice, citationPreferences }) {
  const $ = selector => document.querySelector(selector);
  $("#version-select").addEventListener("change", async (event) => {
    const view = store.get().view; if (!view) return;
    const revision = Number(event.target.value); const epoch = store.beginRequest();
    store.update({ loading: true, view: { ...view, displayRevision: revision, reportSha: null, report: null, renderedReport: {html: "", headings: []}, runtimeEvaluation: {}, quality: null, delivery: null, summary: {...view.summary, quality_status: null, delivery_status: null} } });
    try {
      const version = await api.json(`/survey/${encodeURIComponent(view.identity.runId)}/versions/${revision}?citation_style=${$("#export-style").value}`);
      if (!store.isCurrent(epoch)) return;
      store.update({ loading: false, view: { ...view, displayRevision: revision, reportSha: version.report_sha256, report: version.report, renderedReport: version.rendered_report, runtimeEvaluation: version.report.evaluation || {}, quality: version.report.quality, delivery: version.report.delivery, summary: {...view.summary, quality_status: version.report.quality?.status, delivery_status: version.report.delivery?.status} } });
      setNotice();
    } catch (error) {
      if (!store.isCurrent(epoch)) return;
      store.update({ loading: false }); setNotice(`${t("operationFailed")}: ${error.message}`, "error");
    }
  });
  async function downloadFormat(button, bibliography = false) {
    const view = store.get().view; if (!view) return;
    const revision = Number(view.displayRevision || $("#version-select").value || 1);
    const style = $("#export-style").value;
    const format = bibliography ? button.dataset.bibliography : button.dataset.format;
    const endpoint = bibliography ? "bibliography" : "export";
    button.disabled = true;
    try {
      const response = await api.request(`/survey/${encodeURIComponent(view.identity.runId)}/${endpoint}?revision=${revision}&format=${format}&citation_style=${style}`);
      const blob = await response.blob(); const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url;
      link.download = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] || `litagent.${format}`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
    finally { button.disabled = false; }
  }
  document.querySelectorAll("[data-format]").forEach(button => button.addEventListener("click", () => downloadFormat(button)));
  document.querySelectorAll("[data-bibliography]").forEach(button => button.addEventListener("click", () => downloadFormat(button, true)));
  $("#export-style").addEventListener("change", async event => {
    const view = store.get().view; if (!view) return;
    const style = event.target.value; const revision = Number(view.displayRevision || $("#version-select").value || 1);
    const epoch = store.beginRequest();
    try {
      const path = view.identity.source === "interactive" ? `/survey/${encodeURIComponent(view.identity.runId)}/versions/${revision}` : `/flow-demo/runs/${encodeURIComponent(view.identity.runId)}/view`;
      const raw = await api.json(`${path}?citation_style=${style}`);
      if (!store.isCurrent(epoch)) return;
      citationPreferences.set(view.identity.runId, style);
      store.update({view: {...view, renderedReport: raw.rendered_report}, loading: false});
      setNotice();
    } catch (error) {
      if (!store.isCurrent(epoch)) return;
      store.update({loading: false}); setNotice(`${t("operationFailed")}: ${error.message}`, "error");
    }
  });
}
