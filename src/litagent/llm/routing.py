"""Bind stage models to one shared transport, budget, and trace lifecycle."""

from litagent.llm.client import (
    BaseLLMClient,
    OpenAICompatibleClient,
    ProviderResponseError,
    ResponsesLLMResponse,
)


class ModelBoundClient(BaseLLMClient):
    """Borrow a provider client without owning its HTTP connection or budget."""

    def __init__(self, base, model: str, stage: str):
        """Keep the model binding immutable for the lifetime of the stage view."""
        self._base = base
        self._model = model
        self.stage = stage

    @property
    def _cost_budget(self):
        """Expose the same mutable run budget to production reservation helpers."""
        return self._base._cost_budget

    @property
    def _trace_hook(self):
        """Expose the existing trace hook without creating another root trace."""
        return self._base._trace_hook

    @property
    def _request_timeouts(self):
        """Borrow the transport owner's timeout policy without copying budgets."""
        return getattr(self._base, "_request_timeouts", {})

    def for_model(self, model: str, stage: str):
        """Bind a new role on the same provider and accounting owner."""
        return ModelBoundClient(self._base, model, stage)

    async def chat(self, messages, **kwargs):
        """Reject model substitution and preserve accounting on identity failures."""
        if kwargs.get("model", self._model) != self._model:
            raise ValueError("stage_model_override_conflict")
        response = await self._base.chat(messages, **{**kwargs, "model": self._model})
        if response.model != self._model:
            if self._cost_budget:
                self._cost_budget.stop_after_worker_failure()
            if isinstance(response, ResponsesLLMResponse):
                raise ProviderResponseError("stage_model_identity_mismatch", response)
            raise ValueError("stage_model_identity_mismatch")
        return response

    async def score_ragas_faithfulness(self, **kwargs):
        """Use this view for both calls of the historical RAGAS adapter."""
        return await OpenAICompatibleClient.score_ragas_faithfulness(self, **kwargs)

    async def close(self):
        """Leave transport disposal to its owner."""
        return None


class NativeEvaluationClient:
    """Route native Evaluation calls using the same phase that owns the reservation."""

    def __init__(self, evaluation, inventory_model: str):
        self._evaluation = evaluation
        rebind = getattr(evaluation, "for_model", None)
        self._inventory = (
            rebind(inventory_model, "evaluation_inventory")
            if rebind is not None
            else ModelBoundClient(evaluation, inventory_model, "evaluation_inventory")
        )

    def __getattr__(self, name):
        return getattr(self._evaluation, name)

    async def chat(self, messages, **kwargs):
        """Keep NLI and global review on their original model and verify inventory echoes."""
        from litagent.safety.planned_budget import current_request_stage

        stage = current_request_stage()
        if stage == "inventory":
            return await self._inventory.chat(messages, **kwargs)
        if stage in ("assessment", "global"):
            return await self._evaluation.chat(messages, **kwargs)
        raise ValueError("native_evaluation_request_stage_invalid")
