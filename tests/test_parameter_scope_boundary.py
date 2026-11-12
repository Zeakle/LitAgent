"""Keep parameter types separate from explicit count and scale claims."""

import copy

import pytest

from litagent.eval import relation_scopes
from litagent.eval.relation_scopes import parameter_scope_issue, relation_scope_identity


@pytest.mark.parametrize(
    "text",
    [
        "分类基础 NeRF 时，场景被参数化为连续的 5D 神经场。",
        "The scene is parameterized as a continuous 5D field.",
        "The scene uses a neural parameterization.",
        "The material parameters determine reflectance.",
        "Camera parameters include position and direction.",
        "相机参数包括位置和方向，材质参数包括反射率。",
        "该方法使用网络参数，并说明参数类型。",
        "Its inputs include 5D coordinates and camera parameters.",
        "Parameter-efficient adaptation is considered.",
    ],
)
def test_parameterization_or_parameter_kind_is_not_a_count_scope(text):
    assert parameter_scope_issue(text, []) is None


@pytest.mark.parametrize(
    "text",
    [
        "Model-A increases parameter count.",
        "Model-A reduces parameters.",
        "Model-A needs fewer trainable parameters.",
        "Model-A adds no additional parameters.",
        "Model-A has 8M parameters.",
        "模型参数数量降低。",
        "模型参数量减少。",
        "模型的参数规模更小。",
        "该方法减少参数。",
        "该模型需要8个相机参数。",
        "Model-A reduces model size.",
        "该方法降低模型大小。",
    ],
)
def test_explicit_count_or_size_requires_an_own_quantity_scope(text):
    issue = parameter_scope_issue(text, ["Model-A describes camera parameter types."])
    assert issue is not None
    assert issue["issue"] == "support_relation_unverified"
    assert issue["aspect"] == "parameter_count"


@pytest.mark.parametrize(
    "text,source",
    [
        ("Model-A has 8M parameters.", "Model-A has 8 million parameters."),
        ("Model-A has 8M parameters.", "Model-A的参数量为8百万。"),
        ("模型参数量为8万。", "The number of parameters is 8万."),
        ("模型有8万参数。", "模型有8万参数。"),
        ("模型有8百万参数。", "模型有8百万参数。"),
        (
            "Parameter count is 8 and accuracy is 80%.",
            "Parameter count is 8 and accuracy is 80%.",
        ),
        ("模型需要8个相机参数。", "The model has 8 camera parameters."),
        ("模型的参数规模更小。", "The parameter count is smaller."),
        ("Model-A reduces parameters.", "Model-A reduces parameter count."),
        ("Model-A has fewer parameters.", "Model-A has fewer parameters."),
        ("Model-A reduces model size.", "Model-A reduces model size."),
        ("模型大小为8MB。", "Model size is 8 MB."),
        ("Model-A reduces parameters by 20%.", "Model-A reduces parameters by 20%."),
    ],
)
def test_matching_explicit_scope_remains_subject_to_semantic_host_checks(text, source):
    assert parameter_scope_issue(text, [source]) is None


@pytest.mark.parametrize(
    "text,source",
    [
        ("Model-A has 8M parameters.", "Model-A has 8B parameters."),
        ("Model-A has 8M parameters.", "Model-A has 9M parameters."),
        ("模型参数量为8万。", "模型参数量为8亿。"),
        ("模型需要8个相机参数。", "该模型需要8个材质参数。"),
        ("Model-A has 8M parameters.", "Model-A has 8M pixels and camera parameters."),
        ("Model-A has 8M parameters.", "Its 8M pixels affect parameter count."),
        ("Model-A has 8M parameters.", "Parameter count is unknown; accuracy is 8%."),
        ("Model size is 8MB.", "Model size is 8GB."),
        ("Model size is 8.", "Model size is 8."),
        ("Model-A has 8MB parameters.", "Model-A has 8MB parameters."),
        ("Model-A has 8M parameters.", "Model-A has eight million parameters."),
        ("Model-A has 8M parameters.", "Model-A has 8000K parameters."),
        (
            "Model-A reduces parameters by 20%.",
            "Parameter count is fixed; accuracy falls by 20%.",
        ),
        ("模型大小为8MB。", "模型参数量为8百万。"),
        ("Model-A has 8Z parameters.", "Model-A has 8Z parameters."),
        ("模型有8兆参数。", "模型有8兆参数。"),
        ("模型有8种参数。", "模型有8个参数。"),
        ("Model-A has eight million parameters.", "Model-A has 8M parameters."),
        ("相机参数量为8。", "材质参数量为8。"),
        ("Camera parameter count is 8.", "Material parameter count is 8."),
    ],
)
def test_wrong_parameter_object_value_unit_or_unresolved_scale_is_withheld(
    text, source
):
    issue = parameter_scope_issue(text, [source])
    assert issue is not None and issue["aspect"] == "parameter_count"


def test_scope_does_not_borrow_neighboring_quotes_or_modify_any_input():
    text = "Model-A has 8M parameters."
    quotes = ["Model-A has 8M pixels.", "Camera parameters are listed."]
    before = copy.deepcopy(quotes)
    assert parameter_scope_issue(text, quotes) is not None
    assert quotes == before
    assert text == "Model-A has 8M parameters."


def test_parameterization_has_no_general_scientific_support_result():
    result = parameter_scope_issue("NeRF被参数化为5D场。", [])
    assert result is None
    assert not isinstance(result, dict)


def test_identity_binds_real_helper_source_and_boundary_constants(monkeypatch):
    original = relation_scope_identity()
    assert len(original) == 64 and original == relation_scope_identity()
    monkeypatch.setattr(relation_scopes, "_COUNT_SCOPE", "changed")
    assert relation_scope_identity() != original


def test_identity_binds_the_actual_parser_callable(monkeypatch):
    original = relation_scope_identity()

    def changed_parser(text):
        """Provide a distinct real source for identity drift verification."""
        return []

    monkeypatch.setattr(relation_scopes, "_scope_rows", changed_parser)
    assert relation_scope_identity() != original
