const object = value => value && typeof value === "object" && !Array.isArray(value) ? value : {};
const array = value => Array.isArray(value) ? value : [];
export const evidenceId = value => String(value || "").replace(/^E:/, "");

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(
    Object.keys(value).sort().map(key => [key, canonical(value[key])]),
  );
  return value;
}

function sameSource(left, right) {
  const fields = value => ({
    paper_id: value.paper_id,
    text: value.text ?? value.claim_text,
    supporting_text: value.supporting_text,
    locator: value.source_locator || value.locator || "",
    fact_fields: value.fact_fields || {},
    content_hash: value.content_hash || null,
  });
  return JSON.stringify(canonical(fields(left))) === JSON.stringify(canonical(fields(right)));
}

export function safePaperUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export function probability(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? `${(value * 100).toFixed(1)}%` : "—";
}

// Project recorded decisions only; this reader never reclassifies admission.
export function buildEvidenceLedger(view = {}) {
  const metadata = object(view.report?.metadata);
  const receipt = object(metadata.extraction_verification);
  const candidates = new Map(Object.entries(object(receipt.candidate_ledger))
    .filter(([id, item]) => item && evidenceId(item.evidence_id) === evidenceId(id))
    .map(([id, item]) => [evidenceId(id), item]));
  const selected = new Map(array(view.evidence).filter(item => item?.evidence_id)
    .map(item => [evidenceId(item.evidence_id), item]));
  const catalog = new Map();
  for (const node of Object.values(object(view.graph?.nodes))) {
    for (const paper of array(node?.output)) {
      if (paper?.paper_id && (paper.url || paper.title)) {
        catalog.set(paper.paper_id, { ...catalog.get(paper.paper_id), ...paper });
      }
    }
  }
  const accepted = new Set(array(receipt.accepted_evidence_ids).map(evidenceId));
  const rejected = new Map(Object.entries(object(receipt.rejected)).map(([id, reasons]) => [evidenceId(id), reasons]));
  const inputs = array(receipt.jev?.inputs);
  const decisions = array(receipt.jev?.decisions);
  const pool = new Map([...candidates, ...selected]);
  return [...pool].map(([id, item]) => {
    const candidate = candidates.get(id);
    const bound = candidate && sameSource(candidate, item);
    const inputClaims = new Set(inputs.filter(input =>
      input.source_unit_id === id && array(input.evidence).some(e => evidenceId(e.evidence_id) === id),
    ).map(input => input.claim_id));
    const matched = bound ? decisions.filter(decision => inputClaims.has(decision.claim_id) && (
      (decision.evidence_id != null && evidenceId(decision.evidence_id) === id)
      || (decision.evidence_id == null && decision.kind === "group")),
    ) : [];
    const kinds = new Set(matched.map(decision => decision.kind));
    const complete = matched.length === 2 && new Set(matched.map(d => d.question_id)).size === 2
      && new Set(matched.map(d => d.claim_id)).size === 1
      && matched.every(d => d.status === "valid" && probability(d.confidence) !== "—")
      && ((kinds.has("claim_validity") && kinds.has("claim_format"))
        || (kinds.has("group") && kinds.has("member")));
    let state = "unverified";
    if (candidate && !bound) state = "mismatch";
    else if (bound && ["completed", "degraded"].includes(receipt.status)) {
      if (accepted.has(id) && !rejected.has(id) && complete) state = "accepted";
      else if (rejected.has(id)) state = "rejected";
    } else if (receipt.status === "disabled" || (!candidate && metadata.verification?.jev === false)) state = "disabled";
    const paper = catalog.get(item.paper_id) || {};
    return {
      ...item,
      evidence_id: id,
      paper_title: item.paper_title || paper.title || item.paper_id || "",
      inspection: {
        selected: selected.has(id), state, decisions: matched,
        reasons: bound ? array(rejected.get(id)) : [],
        threshold: bound ? receipt.confidence_threshold ?? null : null,
        receiptStatus: receipt.status || null,
        sourceRunId: bound ? receipt.source_run_id : null,
        model: bound ? receipt.jev?.actual_model || receipt.jev?.requested_model : null,
        url: safePaperUrl(item.url) || safePaperUrl(paper.url)
          || (typeof paper.doi === "string" && /^10\.\d{4,9}\//.test(paper.doi)
            ? `https://doi.org/${encodeURI(paper.doi)}` : null),
      },
    };
  });
}

export function filterEvidence(items, query = "", state = "all") {
  const needle = query.trim().toLocaleLowerCase();
  return items.filter(item => (state === "all" || (state === "selected"
    ? item.inspection?.selected : item.inspection?.state === state))
    && [item.evidence_id, item.paper_id, item.paper_title, item.text, item.supporting_text]
      .some(value => String(value || "").toLocaleLowerCase().includes(needle)));
}
