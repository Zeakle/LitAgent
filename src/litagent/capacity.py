"""Define server-owned capacity limits without changing historical defaults."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY = "adaptive-capacity-v1"
BUDGET = "planned-budget-v6"


class CapacityConfig(BaseModel):
    """Share content bounds and optional operational ceilings across consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    target_body_units: int = Field(default=48, ge=5, le=256)
    max_body_units: int = Field(default=66, ge=5, le=256)
    max_summary_units: int = Field(default=2, ge=1, le=3)
    max_unit_characters: int = Field(default=2000, ge=1, le=2000)
    initial_assertions: int = Field(default=128, ge=1, le=1024)
    batch_size: int = Field(default=16, ge=1, le=16)
    max_assertions: int | None = Field(default=None, ge=1, le=1024)
    max_assessment_batches: int | None = Field(default=None, ge=1, le=64)
    body_json_bytes: int = Field(default=32000, ge=1, le=1000000)
    summary_json_bytes: int = Field(default=4000, ge=1, le=1000000)
    claim_json_bytes: int = Field(default=30000, ge=1, le=1000000)
    claim_max_json_bytes: int = Field(default=2000, ge=1, le=2000)
    diagnostic_json_bytes: int = Field(default=12288, ge=1, le=12288)

    @model_validator(mode="after")
    def validate_dependencies(self):
        """Reject profiles whose guaranteed initial path contradicts their limits."""
        batches = (self.initial_assertions + self.batch_size - 1) // self.batch_size
        if (
            self.target_body_units > self.max_body_units
            or self.max_body_units + 2 * self.max_summary_units
            > self.initial_assertions
            or (
                self.max_assertions is not None
                and self.max_assertions < self.initial_assertions
            )
            or (
                self.max_assessment_batches is not None
                and self.max_assessment_batches < batches
            )
        ):
            raise ValueError("capacity_configuration_conflict")
        return self


def resolve_capacity(value=None):
    """Resolve only server-selected values; callers cannot replace source inputs."""
    if isinstance(value, CapacityConfig):
        return value
    return CapacityConfig.model_validate({} if value is None else value)


def apply_capacity_schema(schema, envelope):
    """Apply the identical server text limit at both request and host boundaries."""
    import copy

    if not envelope or "capacity" not in envelope:
        return schema
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            fields = node.get("properties", {})
            if "text" in fields and (
                "fact_card_refs" in fields or "body_unit_refs" in fields
            ):
                fields["text"]["maxLength"] = envelope["capacity"][
                    "max_unit_characters"
                ]
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result
