const ALLOWED_TAGS = new Set([
  "H1", "H2", "H3", "H4", "H5", "H6", "P", "UL", "OL", "LI", "BLOCKQUOTE",
  "PRE", "CODE", "STRONG", "EM", "S", "HR", "BR", "TABLE", "THEAD", "TBODY",
  "TR", "TH", "TD", "A",
]);

function safeHref(value) {
  if (!value) return null;
  if (value.startsWith("#")) return value;
  try {
    const url = new URL(value, location.origin);
    return ["http:", "https:", "mailto:"].includes(url.protocol) ? value : null;
  } catch {
    return null;
  }
}

function cloneSafe(node, documentRef) {
  if (node.nodeType === Node.TEXT_NODE) return documentRef.createTextNode(node.textContent || "");
  if (node.nodeType !== Node.ELEMENT_NODE || !ALLOWED_TAGS.has(node.tagName)) {
    const fragment = documentRef.createDocumentFragment();
    for (const child of node.childNodes || []) fragment.append(cloneSafe(child, documentRef));
    return fragment;
  }
  const clean = documentRef.createElement(node.tagName.toLowerCase());
  const ordinal = node.getAttribute("data-source-unit-ordinal");
  if (["P", "TD", "TH", "LI"].includes(node.tagName) && /^\d+$/.test(ordinal || "")) clean.setAttribute("data-source-unit-ordinal", ordinal);
  if (/^H[1-6]$/.test(node.tagName) && node.id) clean.id = node.id;
  if (node.tagName === "A") {
    const href = safeHref(node.getAttribute("href"));
    if (href) clean.setAttribute("href", href);
    clean.setAttribute("rel", "noopener noreferrer");
  }
  for (const child of node.childNodes) clean.append(cloneSafe(child, documentRef));
  return clean;
}

function citationNodes(text, evidenceById, documentRef, onEvidence, numbered) {
  const fragment = documentRef.createDocumentFragment();
  const expression = numbered ? /\[(\d+)\]/g : /\[E:([^\]\s,;]+)\]/g;
  let cursor = 0;
  for (const match of text.matchAll(expression)) {
    fragment.append(documentRef.createTextNode(text.slice(cursor, match.index)));
    let evidenceId = `E:${match[1]}`;
    if (numbered) {
      const occurrence = numbered.occurrences[numbered.index];
      if (!occurrence) {
        fragment.append(documentRef.createTextNode(match[0]));
        cursor = match.index + match[0].length;
        continue;
      }
      if (Number(match[1]) === occurrence.number) {
        evidenceId = `E:${occurrence.evidence_id}`;
        numbered.index += 1;
      }
    }
    if (evidenceById.has(evidenceId)) {
      const button = documentRef.createElement("button");
      button.type = "button";
      button.className = "citation-button";
      button.textContent = match[0];
      button.dataset.evidenceId = evidenceId;
      button.setAttribute("aria-label", `Open evidence ${evidenceId}`);
      button.addEventListener("click", () => onEvidence(evidenceId, button));
      fragment.append(button);
    } else {
      const unresolved = documentRef.createElement("span");
      unresolved.className = "citation-unresolved";
      unresolved.textContent = match[0];
      unresolved.title = "Evidence is absent from the current ledger";
      fragment.append(unresolved);
    }
    cursor = match.index + match[0].length;
  }
  fragment.append(documentRef.createTextNode(text.slice(cursor)));
  return fragment;
}

function activateCitations(container, evidenceById, onEvidence, occurrences = []) {
  const numbered = occurrences.length ? { occurrences, index: 0 } : null;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const pattern = numbered ? /\[\d+\]/ : /\[E:[^\]]+\]/;
    if (pattern.test(node.textContent || "") && !node.parentElement?.closest("code, pre, a, button")) targets.push(node);
  }
  for (const node of targets) node.replaceWith(citationNodes(node.textContent || "", evidenceById, document, onEvidence, numbered));
}

export function renderSearchRecords(container, projection, t) {
  if (!container) return;
  container.replaceChildren();
  const heading = document.createElement("h2"); heading.textContent = t("searchRecords"); container.append(heading);
  const record = projection?.methodology;
  if (!record?.html || !record.source_sha256 || record.source_sha256 !== projection?.source_sha256) {
    const empty = document.createElement("p"); empty.textContent = t("searchRecordsUnavailable"); container.append(empty); return;
  }
  const binding = document.createElement("p"); binding.className = "form-help"; binding.textContent = `${t("versionBoundRecords")} · SHA ${record.source_sha256.slice(0, 12)}`; container.append(binding);
  const details = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = t("viewSearchRecords"); details.append(summary);
  const parsed = new DOMParser().parseFromString(record.html, "text/html");
  for (const child of parsed.body.childNodes) details.append(cloneSafe(child, document));
  container.append(details);
}

function collapseAppendix(container, anchor, t) {
  if (!anchor) return;
  const heading = [...container.querySelectorAll("h2")].find(item => item.id === anchor);
  if (!heading || heading.parentElement !== container) return;
  const details = document.createElement("details"); details.className = "report-appendix";
  const summary = document.createElement("summary"); summary.textContent = heading.textContent; details.append(summary);
  heading.before(details);
  let current = heading;
  while (current && (current === heading || current.tagName !== "H2")) {
    const next = current.nextSibling; details.append(current); current = next;
  }
  details.setAttribute("aria-label", t("searchRecords"));
}

export function renderReportDocument(container, projection, evidenceById, t, onEvidence) {
  container.replaceChildren();
  if (!projection?.html) {
    const empty = document.createElement("p");
    empty.className = "empty-state report-empty";
    empty.textContent = t("noReport");
    container.append(empty);
    return { headings: [], unresolvedIds: [] };
  }
  try {
    const parsed = new DOMParser().parseFromString(projection.html, "text/html");
    for (const child of parsed.body.childNodes) container.append(cloneSafe(child, document));
    if (projection.citation_links?.length) {
      for (const link of projection.citation_links) {
        const anchor = container.querySelector(`a[href="#${link.anchor}"]`);
        if (!anchor) continue;
        const button = document.createElement("button"); button.type = "button";
        button.className = "citation-button"; button.textContent = anchor.textContent;
        button.dataset.evidenceIds = JSON.stringify(link.evidence_ids);
        button.setAttribute("aria-label", `Open evidence ${link.evidence_ids.join(", ")}`);
        button.addEventListener("click", () => onEvidence(link.evidence_ids.map(id => `E:${id}`), button));
        anchor.replaceWith(button);
      }
    } else {
      activateCitations(container, evidenceById, onEvidence, projection.citation_occurrences || []);
    }
    collapseAppendix(container, projection.appendix_anchor, t);
  } catch {
    container.textContent = projection.html;
    container.classList.add("render-fallback");
  }
  return { headings: projection.headings || [] };
}

export function renderOutline(container, headings, onNavigate) {
  container.replaceChildren();
  const visible = (headings || []).filter((heading) => heading.level === 2 || heading.level === 3);
  for (const heading of visible) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `outline-item level-${heading.level}`;
    button.textContent = heading.text;
    button.addEventListener("click", () => onNavigate(heading));
    container.append(button);
  }
}
