function action(label, value, className = "") {
  const button = document.createElement("button");
  button.type = "submit";
  button.name = "action";
  button.value = value;
  button.className = className;
  button.textContent = label;
  return button;
}

export function renderReview(container, request, draft, t, onSubmit, onDraftChange) {
  container.replaceChildren();
  if (!request) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = t("noRun");
    container.append(empty);
    return;
  }
  const form = document.createElement("form");
  form.className = "review-form";
  if (request.stage === "paper_selection") {
    const count = document.createElement("output");
    count.className = "selection-count";
    const list = document.createElement("div");
    list.className = "paper-review-list";
    const selected = new Set(draft.selectedPaperIds || request.automatic_selected_paper_ids || []);
    const updateCount = () => { count.textContent = `${selected.size} selected`; };
    for (const paper of request.candidates || []) {
      const label = document.createElement("label");
      label.className = "paper-review-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "paper";
      checkbox.value = paper.paper_id;
      checkbox.checked = selected.has(paper.paper_id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selected.add(paper.paper_id); else selected.delete(paper.paper_id);
        updateCount();
        onDraftChange?.({ ...draft, selectedPaperIds: [...selected] });
      });
      const body = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = paper.title || paper.paper_id;
      const abstract = document.createElement("small");
      abstract.textContent = paper.abstract || paper.source || "";
      body.append(title, abstract);
      label.append(checkbox, body);
      list.append(label);
    }
    updateCount();
    form.append(count, list, action(t("confirm"), "confirm", "primary-command"));
  } else {
    const issues = document.createElement("div");
    issues.className = "review-issues";
    issues.textContent = (request.report?.delivery?.reason_codes || []).join(", ") || request.instructions || "Review the report before delivery.";
    const feedback = document.createElement("textarea");
    feedback.name = "feedback";
    feedback.rows = 7;
    feedback.maxLength = 4000;
    feedback.value = draft.feedback || "";
    feedback.placeholder = t("feedback");
    feedback.addEventListener("input", () => onDraftChange?.({ ...draft, feedback: feedback.value }));
    form.append(issues, feedback);
    let approve = null;
    if ((request.allowed_actions || []).includes("approve")) {
      approve = action(t("approve"), "approve", "primary-command"); form.append(approve);
    }
    if (request.report?.metadata?.quality_policy === "strict_v1") {
      if (request.report_sha256) {
        const binding = document.createElement("p"); binding.className = "audit-binding";
        binding.textContent = `r${request.report_revision ?? request.revision} · SHA ${request.report_sha256}`;
        form.append(binding);
      }
      const checklist = document.createElement("fieldset");
      const legend = document.createElement("legend"); legend.textContent = t("strictChecklist"); checklist.append(legend);
      const inputs = [];
      for (const key of ["checkCoverage", "checkNumbers", "checkComparison", "checkSummary", "checkProposals"]) {
        const label = document.createElement("label"); const input = document.createElement("input"); input.type = "checkbox";
        const text = document.createElement("span"); text.textContent = t(key); label.append(input, text); checklist.append(label); inputs.push(input);
        input.addEventListener("change", () => { if (approve) approve.disabled = !inputs.every(i => i.checked); });
      }
      if (approve) approve.disabled = true;
      else { const note = document.createElement("p"); note.textContent = t("automaticRequired"); checklist.append(note); }
      form.append(checklist);
    }
    if ((request.allowed_actions || []).includes("revise")) form.append(action(t("revise"), "revise"));
    if ((request.allowed_actions || []).includes("reject")) form.append(action(t("reject"), "reject", "danger-command"));
  }
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!event.submitter) return;
    const data = new FormData(form);
    onSubmit(event.submitter.value, {
      selectedPaperIds: data.getAll("paper"),
      feedback: String(data.get("feedback") || ""),
    }, form);
  });
  container.append(form);
}
