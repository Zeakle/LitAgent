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
    meta.textContent = [item.content_scope, item.locator].filter(Boolean).join(" · ");
    button.append(id, title, excerpt, meta);
    button.addEventListener("click", () => onSelect({ kind: "evidence", id: item.evidence_id, value: item }, button));
    article.append(button);
    container.append(article);
  }
}
