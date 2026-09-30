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
    activateCitations(container, evidenceById, onEvidence, projection.citation_occurrences || []);
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
