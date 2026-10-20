"""Bind stage models to one shared transport, budget, and trace lifecycle."""

from litagent.llm.client import BaseLLMClient, OpenAICompatibleClient


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

    async def chat(self, messages, **kwargs):
        """Reject model substitution and preserve accounting on identity failures."""
        if kwargs.get("model", self._model) != self._model:
            raise ValueError("stage_model_override_conflict")
        response = await self._base.chat(messages, **{**kwargs, "model": self._model})
        if response.model != self._model:
            if self._cost_budget:
                self._cost_budget.stop_after_worker_failure()
            raise ValueError("stage_model_identity_mismatch")
        return response

    async def score_ragas_faithfulness(self, **kwargs):
        """Use this view for both calls of the historical RAGAS adapter."""
        return await OpenAICompatibleClient.score_ragas_faithfulness(self, **kwargs)

    async def close(self):
        """Leave transport disposal to its owner."""
        return None
