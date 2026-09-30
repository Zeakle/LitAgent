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
    if ((request.allowed_actions || []).includes("approve")) form.append(action(t("approve"), "approve", "primary-command"));
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
