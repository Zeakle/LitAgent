import { probability } from "./evidence-model.js?v=15.10.6.431";

export function evidenceMap(items = []) {
  return new Map(items.filter((item) => item?.evidence_id).map((item) => {
    const rawId = String(item.evidence_id);
    const citationId = rawId.startsWith("E:") ? rawId : `E:${rawId}`;
    return [citationId, item];
  }));
}

export function renderEvidenceList(container, items, t, onSelect) {
  container.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = t("emptyEvidence");
    container.append(empty);
    return;
  }
  for (const item of items) {
    const article = document.createElement("article");
    article.className = "evidence-row";
    const button = document.createElement("button");
    button.type = "button";
    const id = document.createElement("strong");
    id.textContent = item.evidence_id;
    const title = document.createElement("span");
    title.textContent = item.paper_title || item.paper_id || "Untitled paper";
    const excerpt = document.createElement("p");
    excerpt.textContent = item.text || "";
    const meta = document.createElement("small");
    meta.textContent = [item.content_scope, item.locator || item.source_locator].filter(Boolean).join(" · ");
    button.append(id, title, excerpt, meta);
    if (item.inspection) {
      const badge = document.createElement("span");
      badge.className = "evidence-badge";
      badge.dataset.state = item.inspection.state;
      badge.textContent = `${t(`jev_${item.inspection.state}`)}${item.inspection.selected ? ` · ${t("ledgerSelected")}` : ""}`;
      button.append(badge);
    }
    button.addEventListener("click", () => onSelect({ kind: "evidence", id: item.evidence_id, value: item }, button));
    article.append(button);
    container.append(article);
  }
}

function section(container, title, text) {
  const block = document.createElement("section");
  block.className = "evidence-detail-section";
  const heading = document.createElement("h3"); heading.textContent = title;
  block.append(heading);
  if (text != null) {
    const paragraph = document.createElement("p"); paragraph.textContent = text;
    block.append(paragraph);
  }
  container.append(block);
  return block;
}

export function renderEvidenceInspector(container, selection, t, ledger = []) {
  if (selection?.kind !== "evidence") return false;
  const byId = evidenceMap(ledger);
  if (selection.citationScope?.source_quote) {
    section(container, t("citedReportText"), selection.citationScope.source_quote);
  }
  for (const original of selection.references || [selection.value]) {
    if (!original) continue;
    const id = String(original.evidence_id || selection.id || "").replace(/^E:/, "");
    const item = byId.get(`E:${id}`) || original;
    const inspection = item.inspection || {};
    const source = section(container, item.paper_title || item.paper_id || id, id);
    if (inspection.url) {
      const link = document.createElement("a"); link.href = inspection.url;
      link.target = "_blank"; link.rel = "noopener noreferrer";
      link.textContent = t("openPaper"); source.append(link);
    }
    section(container, t("extractedClaim"), item.text || item.claim_text || t("notRecorded"));
    if (Object.keys(item.fact_fields || {}).length) {
      const fields = section(container, t("extractedFields"));
      const list = document.createElement("dl");
      for (const [key, value] of Object.entries(item.fact_fields)) {
        if (value == null || value === "") continue;
        const term = document.createElement("dt"); term.textContent = key;
        const detail = document.createElement("dd"); detail.textContent = String(value);
        list.append(term, detail);
      }
      fields.append(list);
    }
    const sourceText = section(container, t("verificationSource"), item.supporting_text || t("sourceNotRecorded"));
    const position = document.createElement("small");
    position.textContent = [item.content_scope, item.source_locator || item.locator,
      item.page != null ? `${t("pageNumber")} ${item.page}` : null,
      item.section, item.block_index != null ? `Block ${item.block_index}` : null].filter(Boolean).join(" · ");
    sourceText.append(position);
    const assessment = section(container, t("extractionJev"), t(`jev_${inspection.state || "unverified"}`));
    if (inspection.model) section(assessment, t("auditModel"), inspection.model);
    if (inspection.threshold != null) section(assessment, t("recordedThreshold"), probability(inspection.threshold));
    if (inspection.reasons?.length) section(assessment, t("reasonCodes"), inspection.reasons.join(" · "));
    for (const decision of inspection.decisions || []) {
      const block = section(assessment, t(`jev_${decision.kind}`), `${decision.choice} · ${t("confidence")} ${probability(decision.confidence)}`);
      if (decision.uncertain) section(block, t("uncertain"), t("jevUncertainHelp"));
      for (const [choice, value] of Object.entries(decision.probabilities || {})) {
        const row = document.createElement("div"); row.className = "probability-row";
        const label = document.createElement("span"); label.textContent = choice;
        const score = document.createElement("strong"); score.textContent = probability(value);
        row.append(label, score); block.append(row);
      }
      if (decision.reason_codes?.length) section(block, t("reasonCodes"), decision.reason_codes.join(" · "));
    }
    if (inspection.sourceRunId) section(assessment, t("verificationRun"), inspection.sourceRunId);
    section(container, t("extractorConfidence"), `${probability(item.confidence)} · ${t("extractorConfidenceHelp")}`);
  }
  return true;
}
