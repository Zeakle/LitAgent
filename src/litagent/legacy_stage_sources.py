"""Exact opt-in source deltas excluded from pre-stage-repair fingerprints."""

DELTAS = {
    "WriterSession._call": [
        [
            (
                '                        messages[0]["content"] if view_fields else '
                "system\n"
                "                    ),\n"
                "                )\n"
                "            if stage_repair.enabled(self.envelope) and stage in "
                '{"repair", "summary"}:\n'
                "                category = review_policy.stage_name(self.envelope, "
                'stage).split("/")[0]\n'
                '                if category in {"review_revision", '
                '"evaluation_revision"}:\n'
                "                    bound = {\n"
                '                        "stage": stage,\n'
                '                        "category": category,\n'
                '                        "messages": copy.deepcopy(messages),\n'
                '                        "response_format": copy.deepcopy(rf),\n'
                '                        "schema": copy.deepcopy(schema),\n'
                "                    }\n"
                '                    entry["candidate_request"] = bound\n'
                '                    entry["candidate_request_sha256"] = digest(bound)\n'
                "            reservation = check_grounded_request_budget(\n"
                "                self.llm,\n"
                "                messages,\n",
                '                        messages[0]["content"] if view_fields else '
                "system\n"
                "                    ),\n"
                "                )\n"
                "            reservation = check_grounded_request_budget(\n"
                "                self.llm,\n"
                "                messages,\n",
            )
        ]
    ],
    "WriterSession.messages": [
        [
            (
                "    def messages(self, system, payload, schema):\n"
                '        """Keep legacy bytes unchanged while measuring the complete '
                'new wire."""\n'
                "        if stage_repair.enabled(self.envelope):\n"
                '            feedback = self.journal.get("candidate_feedback", {})\n'
                "            stage = review_policy.stage_name(self.envelope, "
                '"summary")\n'
                '            if stage.split("/", 1)[0] != feedback.get("category"):\n'
                "                feedback = {}\n"
                '            payload = {**payload, "candidate_feedback": feedback}\n'
                "        if self.wire_contract is None:\n"
                "            return request_messages(system, payload, schema)\n"
                "        return request_messages(\n",
                "    def messages(self, system, payload, schema):\n"
                '        """Keep legacy bytes unchanged while measuring the complete '
                'new wire."""\n'
                "        if self.wire_contract is None:\n"
                "            return request_messages(system, payload, schema)\n"
                "        return request_messages(\n",
            )
        ]
    ],
    "WriterSession.patch": [
        [
            (
                '            if cause == "review"\n'
                '            else "evaluation_revision" if cause == "evaluation" else '
                '"capacity"\n'
                "        )\n"
                "        with review_policy.partition(category), "
                "stage_repair.policy(self.envelope):\n"
                "            if stage_repair.enabled(self.envelope) and cause in {\n"
                '                "review",\n'
                '                "evaluation",\n'
                "            }:\n"
                "                resumed = stage_repair.resume_body(\n"
                "                    self, body, targets, category=category\n"
                "                )\n"
                "                if resumed is not None:\n"
                "                    return resumed\n"
                "            return await self._patch(body, targets, cause=cause, "
                "**kwargs)\n",
                '            if cause == "review"\n'
                '            else "evaluation_revision" if cause == "evaluation" else '
                '"capacity"\n'
                "        )\n"
                "        with review_policy.partition(category):\n"
                "            return await self._patch(body, targets, cause=cause, "
                "**kwargs)\n",
            )
        ]
    ],
    "_repair_report": [
        [
            (
                "            except WriterError as exc:\n"
                "                if not capacity_repair:\n"
                "                    summary_rejection = reject_refresh(exc, body)\n"
                "                    retryable = summary_rejection is not None and (\n"
                "                        not stage_repair.enabled(session.envelope)\n"
                '                        or summary_rejection.get("action") != "stop"\n'
                "                    )\n"
                "                raise\n"
                "        try:\n"
                "            candidate, bindings = compile_document(\n",
                "            except WriterError as exc:\n"
                "                if not capacity_repair:\n"
                "                    summary_rejection = reject_refresh(exc, body)\n"
                "                    retryable = summary_rejection is not None\n"
                "                raise\n"
                "        try:\n"
                "            candidate, bindings = compile_document(\n",
            ),
            (
                "        except WriterError as exc:\n"
                "            if not is_summary and not capacity_repair:\n"
                "                summary_rejection = reject_refresh(exc, body, summaries)\n"
                "                retryable = summary_rejection is not None and (\n"
                "                    not stage_repair.enabled(session.envelope)\n"
                '                    or summary_rejection.get("action") != "stop"\n'
                "                )\n"
                "            raise\n"
                "        if (\n"
                "            session.envelope is not None\n",
                "        except WriterError as exc:\n"
                "            if not is_summary and not capacity_repair:\n"
                "                summary_rejection = reject_refresh(exc, body, summaries)\n"
                "                retryable = summary_rejection is not None\n"
                "            raise\n"
                "        if (\n"
                "            session.envelope is not None\n",
            ),
            (
                '                "quality",\n'
                '                [{"candidate_sha256": '
                "hashlib.sha256(candidate.encode()).hexdigest()}],\n"
                "            )\n"
                "        if stage_repair.enabled(session.envelope):\n"
                '            session.journal.pop("pending_candidate", None)\n'
                '            session.journal.pop("candidate_feedback", None)\n'
                "            session.save()\n"
                "        updated_doc = {\n"
                "            **document,\n"
                '            "body": body,\n',
                '                "quality",\n'
                '                [{"candidate_sha256": '
                "hashlib.sha256(candidate.encode()).hexdigest()}],\n"
                "            )\n"
                "        updated_doc = {\n"
                "            **document,\n"
                '            "body": body,\n',
            ),
            (
                "        patch_rejection = None\n"
                "        if (\n"
                "            isinstance(exc, WriterError)\n"
                "            and (\n"
                "                stage_repair.enabled(session.envelope)\n"
                '                or exc.code == "writer_patch_response_mismatch"\n'
                "            )\n"
                "            and inline_citations.selected(session.envelope)\n"
                "            and not capacity_repair\n"
                '            and isinstance(results.get("adversarial_review"), dict)\n',
                "        patch_rejection = None\n"
                "        if (\n"
                "            isinstance(exc, WriterError)\n"
                '            and exc.code == "writer_patch_response_mismatch"\n'
                "            and inline_citations.selected(session.envelope)\n"
                "            and not capacity_repair\n"
                '            and isinstance(results.get("adversarial_review"), dict)\n',
            ),
            (
                '                category="evaluation_revision",\n'
                "            )\n"
                "            if patch_rejection is not None:\n"
                "                retryable = (\n"
                '                    patch_rejection["remaining"] > 0\n'
                '                    and patch_rejection.get("action") != "stop"\n'
                "                )\n"
                '        prior = session.journal.get("failure", {})\n'
                "        failure = classify_failure(exc)\n"
                "        if patch_rejection is not None:\n",
                '                category="evaluation_revision",\n'
                "            )\n"
                "            if patch_rejection is not None:\n"
                '                retryable = patch_rejection["remaining"] > 0\n'
                '        prior = session.journal.get("failure", {})\n'
                "        failure = classify_failure(exc)\n"
                "        if patch_rejection is not None:\n",
            ),
        ]
    ],
    "_segments": [
        [
            (
                "def _segments(text, pattern, *, errors=None, scope_contract=None):\n"
                '    """Yield ordered spans with citations attached only within a local '
                'clause."""\n'
                "    if scope_contract is not None:\n"
                "        yield from _local_segments(text, pattern, scope_contract)\n"
                "        return\n"
                "    from litagent.eval.scoped import _source_boundary_tokens\n"
                "\n"
                "    def reject(code, **fields):\n",
                "def _segments(text, pattern, *, errors=None):\n"
                '    """Yield ordered spans with citations attached only within a local '
                'clause."""\n'
                "    from litagent.eval.scoped import _source_boundary_tokens\n"
                "\n"
                "    def reject(code, **fields):\n",
            )
        ]
    ],
    "annotate": [
        [
            (
                '    """Carry the contract into the sender, parser, and persisted replay '
                'schema."""\n'
                "    if selected(envelope):\n"
                '        schema["x-citation-contract"] = CONTRACT\n'
                "        from litagent.stage_repair import scope\n"
                "\n"
                "        if scope(envelope):\n"
                '            schema["x-citation-scope-contract"] = scope(envelope)\n'
                "    return schema\n",
                '    """Carry the contract into the sender, parser, and persisted replay '
                'schema."""\n'
                "    if selected(envelope):\n"
                '        schema["x-citation-contract"] = CONTRACT\n'
                "    return schema\n",
            )
        ]
    ],
    "check_repair_guidance_record": [
        [
            (
                "@stage_repair.with_policy\n"
                "def check_repair_guidance_record(envelope, record, "
                "**kwargs):\n"
                '    """Authenticate optional metadata without changing the '
                'legacy codec checker."""\n'
                "    prompt_fields = repair_prompt_fields(envelope)\n",
                "def check_repair_guidance_record(envelope, record, "
                "**kwargs):\n"
                '    """Authenticate optional metadata without changing the '
                'legacy codec checker."""\n'
                "    prompt_fields = repair_prompt_fields(envelope)\n",
            )
        ]
    ],
    "compile_document": [
        [
            (
                '                    refs = unit["fact_card_refs"]\n'
                "                if inline:\n"
                "                    try:\n"
                "                        refs = inline_citations.references(\n"
                "                            text, refs, "
                "scope_contract=stage_repair.scope(envelope)\n"
                "                        )\n"
                "                    except WriterError as exc:\n"
                "                        if (\n"
                "                            key in summaries\n",
                '                    refs = unit["fact_card_refs"]\n'
                "                if inline:\n"
                "                    try:\n"
                "                        refs = inline_citations.references(text, refs)\n"
                "                    except WriterError as exc:\n"
                "                        if (\n"
                "                            key in summaries\n",
            ),
            (
                "                )\n"
                "                if inline:\n"
                "                    rendered, local_scopes = "
                "inline_citations.compile_text(\n"
                "                        text,\n"
                "                        evidence_by_ref,\n"
                "                        scope_contract=stage_repair.scope(envelope),\n"
                "                    )\n"
                "                    for index, scope in enumerate(local_scopes):\n"
                "                        citation_scopes.append(\n",
                "                )\n"
                "                if inline:\n"
                "                    rendered, local_scopes = "
                "inline_citations.compile_text(\n"
                "                        text, evidence_by_ref\n"
                "                    )\n"
                "                    for index, scope in enumerate(local_scopes):\n"
                "                        citation_scopes.append(\n",
            ),
            (
                "        bindings.update(\n"
                "            citation_contract=inline_citations.CONTRACT, "
                "citation_scopes=citation_scopes\n"
                "        )\n"
                "        if stage_repair.enabled(envelope):\n"
                '            bindings["citation_scope_contract"] = '
                "stage_repair.SCOPE_CONTRACT\n"
                "        inline_citations.validate_scopes(draft, bindings)\n"
                "    limits = effective_summary_limits(envelope)\n"
                "    if limits is not None:\n",
                "        bindings.update(\n"
                "            citation_contract=inline_citations.CONTRACT, "
                "citation_scopes=citation_scopes\n"
                "        )\n"
                "        inline_citations.validate_scopes(draft, bindings)\n"
                "    limits = effective_summary_limits(envelope)\n"
                "    if limits is not None:\n",
            ),
        ]
    ],
    "compile_text": [
        [
            (
                "def compile_text(text, evidence_by_ref, *, scope_contract=None):\n"
                '    """Return canonical Markdown and its exact local source spans."""\n'
                "    references(text, evidence_by_ref, scope_contract=scope_contract)\n"
                "    chunks, spans, offset = [], [], 0\n"
                "    for start, end, marks in _segments(text, MARK, "
                "scope_contract=scope_contract):\n"
                "        snippet = text[start:end]\n"
                '        rendered = MARK.sub(lambda m: f"[E:{evidence_by_ref[m[1]]}]", '
                "snippet)\n"
                "        left = len(snippet) - len(snippet.lstrip())\n",
                "def compile_text(text, evidence_by_ref):\n"
                '    """Return canonical Markdown and its exact local source spans."""\n'
                "    references(text, evidence_by_ref)\n"
                "    chunks, spans, offset = [], [], 0\n"
                "    for start, end, marks in _segments(text, MARK):\n"
                "        snippet = text[start:end]\n"
                '        rendered = MARK.sub(lambda m: f"[E:{evidence_by_ref[m[1]]}]", '
                "snippet)\n"
                "        left = len(snippet) - len(snippet.lstrip())\n",
            )
        ]
    ],
    "editorial_cycle": [
        [
            (
                "async def editorial_cycle(worker, task, source):\n"
                '    """Review each valid draft, retaining disputes for independent '
                'Evaluation."""\n'
                "    from litagent import stage_repair\n"
                "    from litagent.writing_v2 import (\n"
                "        BODY,\n"
                "        WriterError,\n",
                "async def editorial_cycle(worker, task, source):\n"
                '    """Review each valid draft, retaining disputes for independent '
                'Evaluation."""\n'
                "    from litagent.writing_v2 import (\n"
                "        BODY,\n"
                "        WriterError,\n",
            ),
            (
                "    )\n"
                "    history = []\n"
                '    seen = {current["writing_bindings"]["report_sha256"]}\n'
                "    cached_review = None\n"
                "    for round_number in range(4):\n"
                "        if not stage_repair.enabled(session.envelope) or cached_review is "
                "None:\n"
                "            worker._reviewer._writer_path = session.path\n"
                "            worker._reviewer._writer_run_id = session.run_id\n"
                "            review_task = SubTask(\n"
                "                task_id=task.task_id,\n"
                '                description="Review the current editorial draft",\n'
                '                agent_type="reviewer",\n'
                "                input_data={\n"
                "                    **task.input_data,\n"
                '                    "review_round": round_number,\n'
                '                    "upstream_results": {"synthesis": current},\n'
                "                },\n"
                "            )\n"
                '            async with worker._sub_span(task.task_id, "reviewer", '
                "round_number + 1):\n"
                "                review = await worker._reviewer.execute(review_task)\n"
                "            cached_review = review\n"
                "        else:\n"
                "            review = cached_review\n"
                "        # Reviewer owns the same journal; reopen it before the writer "
                "saves.\n"
                "        session = writer_session(\n"
                "            worker._synthesis._llm,\n",
                "    )\n"
                "    history = []\n"
                '    seen = {current["writing_bindings"]["report_sha256"]}\n'
                "    for round_number in range(4):\n"
                "        worker._reviewer._writer_path = session.path\n"
                "        worker._reviewer._writer_run_id = session.run_id\n"
                "        review_task = SubTask(\n"
                "            task_id=task.task_id,\n"
                '            description="Review the current editorial draft",\n'
                '            agent_type="reviewer",\n'
                "            input_data={\n"
                "                **task.input_data,\n"
                '                "review_round": round_number,\n'
                '                "upstream_results": {"synthesis": current},\n'
                "            },\n"
                "        )\n"
                '        async with worker._sub_span(task.task_id, "reviewer", round_number '
                "+ 1):\n"
                "            review = await worker._reviewer.execute(review_task)\n"
                "        # Reviewer owns the same journal; reopen it before the writer "
                "saves.\n"
                "        session = writer_session(\n"
                "            worker._synthesis._llm,\n",
            ),
            (
                "                        summary_failure=failure,\n"
                "                        summary_rejection=decision,\n"
                "                    )\n"
                "                    if (\n"
                "                        stage_repair.enabled(session.envelope)\n"
                '                        and decision["action"] != "stop"\n'
                "                    ):\n"
                "                        continue\n"
                "                    break\n"
                '            if bindings["report_sha256"] in seen:\n'
                '                row["status"] = "repeated_candidate"\n',
                "                        summary_failure=failure,\n"
                "                        summary_rejection=decision,\n"
                "                    )\n"
                "                    break\n"
                '            if bindings["report_sha256"] in seen:\n'
                '                row["status"] = "repeated_candidate"\n',
            ),
            (
                "            row.update(\n"
                '                status="revised_unverified", '
                'candidate_sha256=bindings["report_sha256"]\n'
                "            )\n"
                "            cached_review = None\n"
                "            if stage_repair.enabled(session.envelope):\n"
                '                session.journal.pop("pending_candidate", None)\n'
                '                session.journal.pop("candidate_feedback", None)\n'
                '            seen.add(bindings["report_sha256"])\n'
                "            current.update(\n"
                "                draft=draft,\n",
                "            row.update(\n"
                '                status="revised_unverified", '
                'candidate_sha256=bindings["report_sha256"]\n'
                "            )\n"
                '            seen.add(bindings["report_sha256"])\n'
                "            current.update(\n"
                "                draft=draft,\n",
            ),
            (
                '                session.journal.setdefault("rejected_failures", {})[\n'
                '                    decision["call"]\n'
                '                ] = session.journal.pop("failure", {})\n'
                "                if (\n"
                "                    stage_repair.enabled(session.envelope)\n"
                '                    and decision["action"] != "stop"\n'
                "                ):\n"
                "                    continue\n"
                "                break\n"
                "            if stage_repair.enabled(session.envelope) and exc.code not in "
                "{\n"
                '                "writer_repair_declined",\n'
                '                "writer_repair_no_progress",\n'
                "            }:\n"
                "                raise\n"
                '            if exc.family != "quality":\n'
                "                if not _settled_rejected_editorial_patch(session, exc):\n"
                "                    raise\n",
                '                session.journal.setdefault("rejected_failures", {})[\n'
                '                    decision["call"]\n'
                '                ] = session.journal.pop("failure", {})\n'
                "                break\n"
                '            if exc.family != "quality":\n'
                "                if not _settled_rejected_editorial_patch(session, exc):\n"
                "                    raise\n",
            ),
        ]
    ],
    "identity": [
        [
            (
                "def identity():\n"
                '    """Bind new request and citation behavior without changing old source '
                'hashes."""\n'
                '    return "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf"\n',
                "def identity():\n"
                '    """Bind new request and citation behavior without changing old source '
                'hashes."""\n'
                "    from pathlib import Path\n"
                "\n"
                "    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n",
            )
        ]
    ],
    "messages": [
        [
            (
                "    )\n"
                "    from litagent.writing_v2 import LIVE_SUMMARY_PROMPT\n"
                "\n"
                '    if plan.get("citation_scope_contract"):\n'
                "        from litagent.stage_repair import feedback_payload\n"
                "\n"
                "        payload = feedback_payload(payload)\n"
                "    original_system = system\n"
                '    if plan["kind"] == "chapter":\n'
                "        system = FINAL_CHAPTER_INSTRUCTION.replace(\n",
                "    )\n"
                "    from litagent.writing_v2 import LIVE_SUMMARY_PROMPT\n"
                "\n"
                "    original_system = system\n"
                '    if plan["kind"] == "chapter":\n'
                "        system = FINAL_CHAPTER_INSTRUCTION.replace(\n",
            ),
            (
                "        for uid, premise in units.items():\n"
                '            text = premise if isinstance(premise, str) else premise.get("text", '
                '"")\n'
                "            if MARK.search(text):\n"
                "                first, mark = (\n"
                "                    uid,\n"
                "                    references(\n"
                '                        text, scope_contract=plan.get("citation_scope_contract")\n'
                "                    )[0],\n"
                "                )\n"
                "                break\n"
                "        example = {\n"
                "            section: [\n",
                "        for uid, premise in units.items():\n"
                '            text = premise if isinstance(premise, str) else premise.get("text", '
                '"")\n'
                "            if MARK.search(text):\n"
                "                first, mark = uid, references(text)[0]\n"
                "                break\n"
                "        example = {\n"
                "            section: [\n",
            ),
            (
                "            system += GROUNDING_INSTRUCTION\n"
                '    elif plan["kind"] == "repair" and system != repair_system(schema):\n'
                '        invalid("writer_inline_repair_prompt_invalid")\n'
                '    if plan.get("citation_scope_contract") and plan["kind"] != "repair":\n'
                "        system = scope_instruction(system)\n"
                "    return [\n"
                '        {"role": "system", "content": system},\n'
                "        {\n",
                "            system += GROUNDING_INSTRUCTION\n"
                '    elif plan["kind"] == "repair" and system != repair_system(schema):\n'
                '        invalid("writer_inline_repair_prompt_invalid")\n'
                "    return [\n"
                '        {"role": "system", "content": system},\n'
                "        {\n",
            ),
        ]
    ],
    "parse": [
        [
            (
                "    )\n"
                "\n"
                "    def target_at(ordinal, source):\n"
                "        if (\n"
                '            plan.get("citation_scope_contract")\n'
                "            and owner_directory is None\n"
                '            and "target_directory" in plan\n'
                "        ):\n"
                '            target_id = source["target_id"]\n'
                '            if target_id not in plan["target_directory"]:\n'
                "                raise w.WriterError(\n"
                '                    "writer_candidate_target_unknown",\n'
                "                    diagnostics=[\n"
                "                        {\n"
                '                            "path": ["patches", ordinal, "target_id"],\n'
                '                            "target_id": target_id,\n'
                '                            "reason": "outside_authorized_directory",\n'
                "                        }\n"
                "                    ],\n"
                "                )\n"
                "        return (\n"
                "            derived_targets[ordinal]\n"
                "            if owner_directory is not None\n",
                "    )\n"
                "\n"
                "    def target_at(ordinal, source):\n"
                "        return (\n"
                "            derived_targets[ordinal]\n"
                "            if owner_directory is not None\n",
            ),
            (
                '        text_span = nodes[(*wire_path, "text")]\n'
                "        marker_spans = []\n"
                "        if inline:\n"
                "            inline_citations.references(\n"
                '                unit["text"], scope_contract=plan.get("citation_scope_contract")\n'
                "            )\n"
                "            marker_spans = inline_citations.raw_occurrences(\n"
                '                raw, unit["text"], text_span, wire_path\n'
                "            )\n",
                '        text_span = nodes[(*wire_path, "text")]\n'
                "        marker_spans = []\n"
                "        if inline:\n"
                '            inline_citations.references(unit["text"])\n'
                "            marker_spans = inline_citations.raw_occurrences(\n"
                '                raw, unit["text"], text_span, wire_path\n'
                "            )\n",
            ),
        ]
    ],
    "patch_rejection_projection": [
        [
            (
                "def patch_rejection_projection(value):\n"
                '    """Expose only authenticated local conflict coordinates and '
                'resource facts."""\n'
                "    import re\n"
                "\n"
                "    from litagent.stage_repair import public_rejection\n"
                "\n"
                "    if (projected := public_rejection(value)) is not None:\n"
                "        return projected\n"
                "\n"
                "    try:\n"
                '        if value["contract"] != PATCH_REJECTION_CONTRACT:\n',
                "def patch_rejection_projection(value):\n"
                '    """Expose only authenticated local conflict coordinates and '
                'resource facts."""\n'
                "    import re\n"
                "\n"
                "    try:\n"
                '        if value["contract"] != PATCH_REJECTION_CONTRACT:\n',
            )
        ]
    ],
    "position_diagnostics": [
        [
            (
                "def position_diagnostics(text, *, scope_contract=None):\n"
                '    """Collect syntax-position findings for a rejected candidate '
                'only."""\n'
                '    if MARK_START.search(MARK.sub("", text)) or '
                "EVIDENCE.search(text):\n"
                '        invalid("writer_inline_citation_syntax_invalid")\n'
                "    errors = []\n"
                "    for start, end, marks in _segments(\n"
                "        text, MARK, errors=errors, scope_contract=scope_contract\n"
                "    ):\n"
                '        if marks and not MARK.sub("", text[start:end]).strip(\n'
                '            " \\t。！？.!?；;，," + ("、" if scope_contract else "")\n'
                "        ):\n"
                "            errors.append(\n"
                '                {"code": "writer_inline_citation_without_prose", '
                '"offset": start}\n'
                "            )\n",
                "def position_diagnostics(text):\n"
                '    """Collect syntax-position findings for a rejected candidate '
                'only."""\n'
                '    if MARK_START.search(MARK.sub("", text)) or '
                "EVIDENCE.search(text):\n"
                '        invalid("writer_inline_citation_syntax_invalid")\n'
                "    errors = []\n"
                "    for start, end, marks in _segments(text, MARK, errors=errors):\n"
                '        if marks and not MARK.sub("", text[start:end]).strip(" '
                '\\t。！？.!?；;，,"):\n'
                "            errors.append(\n"
                '                {"code": "writer_inline_citation_without_prose", '
                '"offset": start}\n'
                "            )\n",
            )
        ]
    ],
    "prepare": [
        [
            (
                '        if schema.get("x-citation-contract") == inline_citations.CONTRACT:\n'
                '            plan["wire_schema"] = '
                'inline_citations.wire_schema(plan["wire_schema"])\n'
                '            plan["inline_citations"] = True\n'
                '            if schema.get("x-citation-scope-contract"):\n'
                '                plan["citation_scope_contract"] = '
                'schema["x-citation-scope-contract"]\n'
                "        return plan\n"
                "    except (KeyError, TypeError, SchemaError, AttributeError) as exc:\n"
                '        raise _writer().WriterError("writer_flat_contract_invalid") from exc\n',
                '        if schema.get("x-citation-contract") == inline_citations.CONTRACT:\n'
                '            plan["wire_schema"] = '
                'inline_citations.wire_schema(plan["wire_schema"])\n'
                '            plan["inline_citations"] = True\n'
                "        return plan\n"
                "    except (KeyError, TypeError, SchemaError, AttributeError) as exc:\n"
                '        raise _writer().WriterError("writer_flat_contract_invalid") from exc\n',
            )
        ]
    ],
    "project_structured_repair_guidance": [
        [
            (
                "                _, ranges = "
                "inline_citations.compile_text(\n"
                '                    unit["text"],\n'
                '                    {ref: cards[ref]["evidence_id"] for '
                'ref in unit["fact_card_refs"]},\n'
                "                    "
                "scope_contract=stage_repair.current_scope(),\n"
                "                )\n"
                "                if (\n"
                "                    not 0 <= index < len(ranges)\n",
                "                _, ranges = "
                "inline_citations.compile_text(\n"
                '                    unit["text"],\n'
                '                    {ref: cards[ref]["evidence_id"] for '
                'ref in unit["fact_card_refs"]},\n'
                "                )\n"
                "                if (\n"
                "                    not 0 <= index < len(ranges)\n",
            )
        ]
    ],
    "references": [
        [
            (
                "def references(text, allowed=None, *, scope_contract=None):\n"
                '    """Validate short marks and return their first occurrence order."""\n'
                '    if MARK_START.search(MARK.sub("", text)) or EVIDENCE.search(text):\n'
                '        invalid("writer_inline_citation_syntax_invalid")\n'
                "    refs = []\n"
                "    for start, end, marks in _segments(text, MARK, "
                "scope_contract=scope_contract):\n"
                '        if marks and not MARK.sub("", text[start:end]).strip(\n'
                '            " \\t。！？.!?；;，," + ("、" if scope_contract else "")\n'
                "        ):\n"
                '            invalid("writer_inline_citation_without_prose", offset=start)\n'
                "        for mark in marks:\n"
                "            ref = mark[1]\n",
                "def references(text, allowed=None):\n"
                '    """Validate short marks and return their first occurrence order."""\n'
                '    if MARK_START.search(MARK.sub("", text)) or EVIDENCE.search(text):\n'
                '        invalid("writer_inline_citation_syntax_invalid")\n'
                "    refs = []\n"
                "    for start, end, marks in _segments(text, MARK):\n"
                '        if marks and not MARK.sub("", text[start:end]).strip(" '
                '\\t。！？.!?；;，,"):\n'
                '            invalid("writer_inline_citation_without_prose", offset=start)\n'
                "        for mark in marks:\n"
                "            ref = mark[1]\n",
            )
        ]
    ],
    "reject_patch_candidate": [
        [
            (
                "    import base64\n"
                "    import hashlib\n"
                "\n"
                "    from litagent import stage_repair\n"
                "    from litagent.inline_citations import selected\n"
                "    from litagent.llm.responses import (\n"
                "        build_responses_request,\n",
                "    import base64\n"
                "    import hashlib\n"
                "\n"
                "    from litagent.inline_citations import selected\n"
                "    from litagent.llm.responses import (\n"
                "        build_responses_request,\n",
            ),
            (
                "    )\n"
                "    from litagent.safety.planned_budget import planned_backend\n"
                "    from litagent.writing_v2 import WriterError, compile_document, "
                "review_payload\n"
                "\n"
                "    if stage_repair.enabled(session.envelope):\n"
                "        return stage_repair.reject_candidate(\n"
                "            session, error, retained, category=category\n"
                "        )\n"
                "\n"
                "    if not selected(session.envelope) or error.code != "
                '"writer_patch_response_mismatch":\n'
                "        return None\n",
                "    )\n"
                "    from litagent.safety.planned_budget import planned_backend\n"
                "    from litagent.writing_v2 import WriterError, compile_document, "
                "review_payload\n"
                "\n"
                "    if not selected(session.envelope) or error.code != "
                '"writer_patch_response_mismatch":\n'
                "        return None\n",
            ),
        ]
    ],
    "reject_summary_candidate": [
        [
            (
                "def reject_summary_candidate(session, error, body, summaries, "
                "retained, *, category):\n"
                '    """Authenticate rejection and independently recompile the '
                'retained manuscript."""\n'
                "    from litagent import stage_repair\n"
                "    from litagent.writing_v2 import compile_document, "
                "patch_provenance, review_payload\n"
                "\n"
                "    if stage_repair.enabled(session.envelope):\n"
                "        return stage_repair.reject_candidate(\n"
                "            session, error, retained, category=category, "
                "body=body, summaries=summaries\n"
                "        )\n"
                "\n"
                "    proof = settled_rejected_summary(session, error, body, "
                "summaries, category=category)\n"
                "    if not proof:\n",
                "def reject_summary_candidate(session, error, body, summaries, "
                "retained, *, category):\n"
                '    """Authenticate rejection and independently recompile the '
                'retained manuscript."""\n'
                "    from litagent.writing_v2 import compile_document, "
                "patch_provenance, review_payload\n"
                "\n"
                "    proof = settled_rejected_summary(session, error, body, "
                "summaries, category=category)\n"
                "    if not proof:\n",
            )
        ]
    ],
    "repair_request_input": [
        [
            (
                "@stage_repair.with_policy\n"
                "def repair_request_input(\n"
                "    packet,\n"
                "    body,\n",
                "def repair_request_input(\n    packet,\n    body,\n",
            )
        ]
    ],
    "repair_system": [
        [
            (
                '        "use its short tN key and the schema\'s reference field.",\n'
                '        "use its short tN key and the fields in the schema.",\n'
                "    )\n"
                "    result = (\n"
                "        base\n"
                "        + INSTRUCTION\n"
                '        + "\\nDisposition example: if t1=replace, t2=keep, '
                't3=cannot_resolve, "\n',
                '        "use its short tN key and the schema\'s reference field.",\n'
                '        "use its short tN key and the fields in the schema.",\n'
                "    )\n"
                "    return (\n"
                "        base\n"
                "        + INSTRUCTION\n"
                '        + "\\nDisposition example: if t1=replace, t2=keep, '
                't3=cannot_resolve, "\n',
            ),
            (
                '        + _repair_prompt_type_map(prepare(schema)["wire_schema"])\n'
                '        + ".\\n"\n'
                "    )\n"
                "    return (\n"
                '        scope_instruction(result) if schema.get("x-citation-scope-contract") '
                "else result\n"
                "    )\n",
                '        + _repair_prompt_type_map(prepare(schema)["wire_schema"])\n'
                '        + ".\\n"\n'
                "    )\n",
            ),
        ]
    ],
    "summary_rejection_projection": [
        [
            (
                "def summary_rejection_projection(value):\n"
                '    """Whitelist local rejection facts without exposing '
                'receipts or report text."""\n'
                "    import re\n"
                "\n"
                "    from litagent.stage_repair import public_rejection\n"
                "\n"
                "    if (projected := public_rejection(value)) is not None:\n"
                "        return projected\n"
                "\n"
                "    if (\n"
                "        not isinstance(value, dict)\n",
                "def summary_rejection_projection(value):\n"
                '    """Whitelist local rejection facts without exposing '
                'receipts or report text."""\n'
                "    import re\n"
                "\n"
                "    if (\n"
                "        not isinstance(value, dict)\n",
            )
        ]
    ],
    "validate_body": [
        [
            (
                '        _valid_text(unit["text"])\n'
                "        if inline_citations.selected(envelope):\n"
                "            if (\n"
                "                inline_citations.references(\n"
                '                    unit["text"], aliases, '
                "scope_contract=stage_repair.scope(envelope)\n"
                "                )\n"
                '                != unit["fact_card_refs"]\n'
                "            ):\n"
                "                raise "
                'WriterError("writer_inline_citation_binding_invalid")\n',
                '        _valid_text(unit["text"])\n'
                "        if inline_citations.selected(envelope):\n"
                "            if (\n"
                '                inline_citations.references(unit["text"], aliases)\n'
                '                != unit["fact_card_refs"]\n'
                "            ):\n"
                "                raise "
                'WriterError("writer_inline_citation_binding_invalid")\n',
            )
        ]
    ],
    "validate_scopes": [
        [
            (
                '        start, end = unit["source_start"], unit["source_end"]\n'
                '        actual = by_unit.pop(unit["unit_id"], [])\n'
                "        expected = []\n"
                "        for left, right, _ in _segments(\n"
                "            survey[start:end],\n"
                "            EVIDENCE,\n"
                '            scope_contract=bindings.get("citation_scope_contract"),\n'
                "        ):\n"
                "            text = survey[start + left : start + right]\n"
                "            a = start + left + len(text) - len(text.lstrip())\n"
                "            b = start + left + len(text.rstrip())\n",
                '        start, end = unit["source_start"], unit["source_end"]\n'
                '        actual = by_unit.pop(unit["unit_id"], [])\n'
                "        expected = []\n"
                "        for left, right, _ in _segments(survey[start:end], EVIDENCE):\n"
                "            text = survey[start + left : start + right]\n"
                "            a = start + left + len(text) - len(text.lstrip())\n"
                "            b = start + left + len(text.rstrip())\n",
            )
        ]
    ],
}


# Exact opt-in capacity additions; restore the preceding stage policy first.
CAPACITY_DELTAS = {
    "WriterSession._call": [
        (
            '                if category in {"review_revision", '
            '"evaluation_revision", "capacity"}:\n',
            '                if category in {"review_revision", '
            '"evaluation_revision"}:\n',
        )
    ],
    "WriterSession._patch": [
        (
            "        if inline_citations.selected(self.envelope) and (\n"
            '            cause in {"review", "evaluation"}\n'
            '            or (cause == "capacity" and '
            "stage_repair.enabled(self.envelope))\n"
            "        ):\n",
            "        if inline_citations.selected(self.envelope) and "
            "cause in {\n"
            '            "review",\n'
            '            "evaluation",\n'
            "        }:\n",
        )
    ],
    "_repair_report": [
        (
            "        if (\n"
            "            isinstance(exc, WriterError)\n"
            "            and capacity_repair\n"
            "            and stage_repair.enabled(session.envelope)\n"
            '            and isinstance(results.get("adversarial_review"), '
            "dict)\n"
            "        ):\n"
            "            # Capacity owns its own settled attempt and "
            "authorization directory.\n"
            "            # A rejected response may be retried; the retained "
            "report is unchanged.\n"
            "            patch_rejection = stage_repair.reject_candidate(\n"
            "                session,\n"
            "                exc,\n"
            "                {\n"
            '                    **results["adversarial_review"],\n'
            "                    **meta,\n"
            '                    "draft": original,\n'
            '                    "report_language": report_language,\n'
            "                },\n"
            '                category="capacity",\n'
            "            )\n"
            "            retryable = patch_rejection is not None and (\n"
            '                patch_rejection["action"] != "stop"\n'
            "            )\n",
            "",
        )
    ],
    "parse": [
        (
            "    if owner_directory is not None and "
            'plan.get("citation_scope_contract"):\n'
            "        from litagent.stage_repair import source_owner_diagnostics\n"
            "\n"
            "        # The host directory was validated by prepare(); these findings "
            "concern\n"
            "        # only the returned candidate, never authorization or source "
            "integrity.\n"
            '        findings = source_owner_diagnostics(value["patches"], '
            "owner_directory)\n"
            "        if findings:\n"
            "            raise w.WriterError(\n"
            '                "writer_capacity_source_map_invalid", '
            "diagnostics=findings\n"
            "            )\n",
            "",
        )
    ],
}


PRIOR_FEEDBACK_DELTAS = {
    "check_guidance_wire_record": [
        (
            "                targets if targets is not None else "
            'record["authorized_targets"],\n'
            "                _guidance_depth=depth + 1,\n"
            "            )\n"
            "            replay_failure = "
            'witness["previous_failure"]\n'
            "            if stage_repair.enabled(envelope):\n"
            "                selected = "
            "stage_repair.select_prior_feedback(fresh, "
            "replay_failure)\n"
            "                fresh, replay_failure = "
            'selected["directory"], selected["failure"]\n'
            "                if fresh is None or digest(\n"
            "                    "
            'record["arguments"]["previous_failure"]\n'
            "                ) != digest(replay_failure):\n"
            '                    raise ValueError("guidance prior '
            'selection")\n'
            "            if digest(fresh) != "
            'digest(authority["feedback_context"]):\n'
            '                raise ValueError("guidance prior '
            'directory")\n'
            "            if digest(\n"
            "                prior_repair_guidance_context(\n"
            "                    fresh,\n"
            '                    witness["prior"]["bindings"],\n'
            '                    replay_failure["diagnostics"],\n'
            "                )\n"
            "            ) != "
            'digest(authority["guidance_context"]["prior_candidate"]):\n'
            '                raise ValueError("guidance prior '
            'context")\n',
            "                targets if targets is not None else "
            'record["authorized_targets"],\n'
            "                _guidance_depth=depth + 1,\n"
            "            )\n"
            "            if digest(fresh) != "
            'digest(authority["feedback_context"]):\n'
            '                raise ValueError("guidance prior '
            'directory")\n'
            "            if digest(\n"
            "                prior_repair_guidance_context(\n"
            "                    fresh,\n"
            '                    witness["prior"]["bindings"],\n'
            "                    "
            'witness["previous_failure"]["diagnostics"],\n'
            "                )\n"
            "            ) != "
            'digest(authority["guidance_context"]["prior_candidate"]):\n'
            '                raise ValueError("guidance prior '
            'context")\n',
        )
    ],
    "WriterSession._patch": [
        (
            "        feedback_context=None,\n"
            "        guidance_context=None,\n"
            "        guidance_replay_witness=None,\n"
            "        prior_feedback_failure=None,\n"
            "    ):\n"
            '        """Repair only authorized units within the shared '
            'two-attempt limit."""\n'
            "        if self.writing_plan == RANGE_WRITING_PLAN:\n",
            "        feedback_context=None,\n"
            "        guidance_context=None,\n"
            "        guidance_replay_witness=None,\n"
            "    ):\n"
            '        """Repair only authorized units within the shared '
            'two-attempt limit."""\n'
            "        if self.writing_plan == RANGE_WRITING_PLAN:\n",
        ),
        (
            "                    copy.deepcopy(guidance_context)\n"
            "                )\n"
            "                self.save()\n"
            '            request_failure = self.journal.get("failure")\n'
            "            if prior_feedback_failure is not None:\n"
            "                if (\n"
            "                    not "
            "stage_repair.enabled(self.envelope)\n"
            "                    or guidance_replay_witness is None\n"
            "                    or feedback_context is None\n"
            "                    or "
            'digest(guidance_replay_witness["previous_failure"])\n'
            "                    != digest(request_failure)\n"
            "                ):\n"
            "                    raise "
            'WriterError("writer_feedback_identity_invalid", '
            '"identity")\n'
            "                request_failure = prior_feedback_failure\n"
            "            captured_guidance = []\n"
            "            system, payload, schema = "
            "repair_request_input(\n"
            "                self.packet,\n",
            "                    copy.deepcopy(guidance_context)\n"
            "                )\n"
            "                self.save()\n"
            "            captured_guidance = []\n"
            "            system, payload, schema = "
            "repair_request_input(\n"
            "                self.packet,\n",
        ),
        (
            "                targets,\n"
            "                references,\n"
            "                cause=cause,\n"
            "                previous_failure=request_failure,\n"
            "                writing_plan=self.writing_plan,\n"
            "                envelope=self.envelope,\n"
            "                wire_contract=self.wire_contract,\n",
            "                targets,\n"
            "                references,\n"
            "                cause=cause,\n"
            "                "
            'previous_failure=self.journal.get("failure"),\n'
            "                writing_plan=self.writing_plan,\n"
            "                envelope=self.envelope,\n"
            "                wire_contract=self.wire_contract,\n",
        ),
    ],
    "_repair_report": [
        (
            '            raise WriterError("writer_no_editable_targets", '
            '"quality")\n'
            "        feedback_context = None\n"
            "        deferred_feedback = None\n"
            "        prior_feedback_failure = None\n"
            '        previous_failure = session.journal.get("failure") or {}\n'
            "        if repetition_base or (\n"
            "            review_policy.enabled(session.envelope)\n",
            '            raise WriterError("writer_no_editable_targets", '
            '"quality")\n'
            "        feedback_context = None\n"
            "        deferred_feedback = None\n"
            '        previous_failure = session.journal.get("failure") or {}\n'
            "        if repetition_base or (\n"
            "            review_policy.enabled(session.envelope)\n",
        ),
        (
            "                previous_failure,\n"
            "                targets,\n"
            "            )\n"
            "            if stage_repair.enabled(session.envelope) and "
            "(recovery or {}).get(\n"
            '                "action"\n'
            '            ) in ("prose", "combined"):\n'
            "                selected = stage_repair.select_prior_feedback(\n"
            "                    feedback_context, previous_failure\n"
            "                )\n"
            '                feedback_context = selected["directory"]\n'
            '                prior_feedback_failure = selected["failure"]\n'
            '                deferred_feedback = selected["audit"]\n'
            "            else:\n"
            "                deferred_feedback = "
            "defer_disjoint_prior_feedback(\n"
            "                    session.envelope,\n"
            "                    feedback_context,\n"
            "                    previous_failure,\n"
            '                    action=(recovery or {}).get("action"),\n'
            "                )\n"
            '            if (recovery or {}).get("action") == "capacity":\n'
            "                from litagent.eval.native_capacity import "
            "retry_feedback\n"
            "\n"
            "                deferred_feedback = retry_feedback(\n"
            "                    previous_failure, prior[0], "
            "feedback_context\n"
            "                )\n"
            "            if deferred_feedback is not None and "
            "prior_feedback_failure is None:\n"
            "                feedback_context = None\n"
            "        current_capacity_instruction = None\n"
            "        if capacity_repair and "
            'session.envelope.get("repair_mapping_contract") in (\n',
            "                previous_failure,\n"
            "                targets,\n"
            "            )\n"
            "            deferred_feedback = defer_disjoint_prior_feedback(\n"
            "                session.envelope,\n"
            "                feedback_context,\n"
            "                previous_failure,\n"
            '                action=(recovery or {}).get("action"),\n'
            "            )\n"
            '            if (recovery or {}).get("action") == "capacity":\n'
            "                from litagent.eval.native_capacity import "
            "retry_feedback\n"
            "\n"
            "                deferred_feedback = retry_feedback(\n"
            "                    previous_failure, prior[0], "
            "feedback_context\n"
            "                )\n"
            "            if deferred_feedback is not None:\n"
            "                feedback_context = None\n"
            "        current_capacity_instruction = None\n"
            "        if capacity_repair and "
            'session.envelope.get("repair_mapping_contract") in (\n',
        ),
        (
            "                prior_context = prior_repair_guidance_context(\n"
            "                    feedback_context,\n"
            '                    prior[0]["bindings"],\n'
            "                    (prior_feedback_failure or "
            'previous_failure)["diagnostics"],\n'
            "                )\n"
            "            guidance_context = {\n"
            '                "current": current_context,\n'
            '                "prior_candidate": prior_context,\n'
            "            }\n"
            "            if deferred_feedback is not None:\n"
            "                # Retain complete prior findings locally, "
            "including those that\n"
            "                # cannot authorize edits to the selected working "
            "report.\n"
            "                "
            'session.journal.setdefault("deferred_prior_feedback", '
            "[]).append(\n"
            "                    deferred_feedback\n"
            "                )\n"
            "                if prior_feedback_failure is None:\n"
            '                    session.journal["failure"] = {\n'
            '                        "code": (\n'
            "                            "
            '"native_capacity_candidate_still_oversized"\n'
            '                            if (recovery or {}).get("action") == '
            '"capacity"\n'
            "                            else "
            '"writer_prior_feedback_outside_targets"\n'
            "                        ),\n"
            '                        "family": "capacity" if capacity_repair '
            'else "quality",\n'
            "                    }\n"
            "                session.save()\n"
            "        wire_replay_options = {}\n"
            "        if guidance_wire_fields(session.envelope) and "
            "feedback_context is not None:\n",
            "                prior_context = prior_repair_guidance_context(\n"
            "                    feedback_context,\n"
            '                    prior[0]["bindings"],\n'
            '                    previous_failure["diagnostics"],\n'
            "                )\n"
            "            guidance_context = {\n"
            '                "current": current_context,\n'
            '                "prior_candidate": prior_context,\n'
            "            }\n"
            "            if deferred_feedback is not None:\n"
            "                # The failed candidate remains in the journal. "
            "The provider\n"
            "                # receives only the original report's authorized "
            "diagnoses.\n"
            "                "
            'session.journal.setdefault("deferred_prior_feedback", '
            "[]).append(\n"
            "                    deferred_feedback\n"
            "                )\n"
            '                session.journal["failure"] = {\n'
            '                    "code": (\n'
            "                        "
            '"native_capacity_candidate_still_oversized"\n'
            '                        if (recovery or {}).get("action") == '
            '"capacity"\n'
            "                        else "
            '"writer_prior_feedback_outside_targets"\n'
            "                    ),\n"
            '                    "family": "capacity" if capacity_repair else '
            '"quality",\n'
            "                }\n"
            "                session.save()\n"
            "        wire_replay_options = {}\n"
            "        if guidance_wire_fields(session.envelope) and "
            "feedback_context is not None:\n",
        ),
        (
            '                "prior": copy.deepcopy(prior[0]),\n'
            '                "previous_failure": '
            "copy.deepcopy(previous_failure),\n"
            "            }\n"
            "            if prior_feedback_failure is not None:\n"
            '                wire_replay_options["prior_feedback_failure"] = '
            "prior_feedback_failure\n"
            '        is_summary = next(iter(targets.values()))["path"][0] in '
            "{\n"
            '            "abstract",\n'
            '            "conclusion",\n',
            '                "prior": copy.deepcopy(prior[0]),\n'
            '                "previous_failure": '
            "copy.deepcopy(previous_failure),\n"
            "            }\n"
            '        is_summary = next(iter(targets.values()))["path"][0] in '
            "{\n"
            '            "abstract",\n'
            '            "conclusion",\n',
        ),
    ],
}


def restore_source(name, source):
    """Recover a known legacy source view without changing executed code."""
    for new, old in PRIOR_FEEDBACK_DELTAS.get(name, []):
        if source.count(new) != 1:
            raise ValueError("prior_feedback_legacy_source_projection_invalid")
        source = source.replace(new, old, 1)
    for new, old in CAPACITY_DELTAS.get(name, []):
        if source.count(new) != 1:
            raise ValueError("capacity_legacy_source_projection_invalid")
        source = source.replace(new, old, 1)
    for variant in DELTAS.get(name, []):
        if all(source.count(new) == 1 for new, old in variant):
            for new, old in variant:
                source = source.replace(new, old, 1)
            return source
    return source
