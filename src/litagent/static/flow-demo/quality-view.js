function scoreText(result) {
  if (result?.score == null) return "—";
  const numeric = Number(result.score);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "—";
}

export function renderQuality(container, evaluation, quality, delivery, onSelect) {
  container.replaceChildren();
  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  for (const [name, result] of Object.entries(evaluation || {})) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "metric-row";
    button.dataset.status = result?.passed === true ? "passed" : result?.passed === false ? "failed" : "unverified";
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    const score = document.createElement("strong");
    score.textContent = scoreText(result);
    const status = document.createElement("small");
    status.textContent = button.dataset.status;
    button.append(label, score, status);
    button.addEventListener("click", () => onSelect({ kind: "metric", id: name, value: result }, button));
    metrics.append(button);
  }
  const gate = document.createElement("section");
  gate.className = "quality-gate";
  const title = document.createElement("h3");
  title.textContent = `${quality?.status || "unverified"} / ${delivery?.status || "pending"}`;
  const reasons = document.createElement("p");
  reasons.textContent = (delivery?.reason_codes || []).join(", ") || "No degradation reason recorded.";
  gate.append(title, reasons);
  container.append(metrics, gate);
}
