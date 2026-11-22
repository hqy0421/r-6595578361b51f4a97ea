"""Tests for column lineage and refit impact analysis."""

import pytest

from tubular.comparison import WhenThenOtherwiseTransformer
from tubular.imputers import MeanImputer
from tubular.lineage import (
    analyze_impact,
    build_lineage,
    get_input_columns,
    get_output_columns,
    step_requires_fit,
)
from tubular.misc import ColumnDtypeSetter, RenameColumnsTransformer
from tubular.numeric import DifferenceTransformer, OneDKmeansTransformer
from tubular.strings import StringContainsTransformer


class TestGetInputColumns:
    def test_direct_columns_only(self):
        transformer = MeanImputer(columns=["a", "b"])

        assert get_input_columns(transformer) == ["a", "b"]

    def test_weights_column_indirect_reference(self):
        transformer = MeanImputer(columns=["a"], weights_column="w")

        assert get_input_columns(transformer) == ["a", "w"]

    def test_when_then_otherwise_indirect_references(self):
        transformer = WhenThenOtherwiseTransformer(
            columns=["a"],
            when_column="flag",
            then_column="value",
        )

        assert get_input_columns(transformer) == ["a", "flag", "value"]

    def test_reference_not_a_column_by_default(self):
        transformer = StringContainsTransformer(columns=["a"], reference="x")

        assert get_input_columns(transformer) == ["a"]

    def test_reference_as_column_included_when_flag_set(self):
        transformer = StringContainsTransformer(
            columns=["a"],
            reference="other",
            reference_as_column=True,
        )

        assert get_input_columns(transformer) == ["a", "other"]

    def test_deduplicated_and_sorted(self):
        transformer = MeanImputer(
            columns=["b", "a", "a"],
            weights_column="a",
        )

        assert get_input_columns(transformer) == ["a", "b"]


class TestGetOutputColumns:
    def test_default_outputs_are_columns(self):
        transformer = MeanImputer(columns=["a", "b"])

        assert get_output_columns(transformer) == ["a", "b"]

    def test_created_column_output(self):
        transformer = DifferenceTransformer(columns=["a", "b"])

        assert get_output_columns(transformer) == ["a_minus_b"]


class TestStepRequiresFit:
    def test_fitting_transformer(self):
        assert step_requires_fit(MeanImputer(columns="a")) is True

    def test_stateless_transformer_fits_false(self):
        assert step_requires_fit(DifferenceTransformer(columns=["a", "b"])) is False

    def test_stateless_transformer_in_init_fitted(self):
        assert (
            step_requires_fit(ColumnDtypeSetter(columns="a", dtype="Float64")) is False
        )


class TestBuildLineage:
    def test_root_fields_resolved_through_indirect_output(self):
        steps = [
            ("diff", DifferenceTransformer(columns=["a", "b"])),
            (
                "kmeans",
                OneDKmeansTransformer(
                    columns="a_minus_b",
                    new_column_name="cluster",
                    n_clusters=2,
                ),
            ),
        ]

        lineage = build_lineage(steps)

        assert lineage["diff"].input_root_fields == {"a": ["a"], "b": ["b"]}
        assert lineage["diff"].output_root_fields == {
            "a_minus_b": ["a", "b"],
        }
        assert lineage["kmeans"].input_root_fields == {
            "a_minus_b": ["a", "b"],
        }
        assert lineage["kmeans"].output_root_fields == {"cluster": ["a", "b"]}
        assert lineage["kmeans"].producer_step == {"a_minus_b": ["diff"]}

    def test_renamed_column_traces_to_real_root_field(self):
        steps = [
            (
                "rename",
                RenameColumnsTransformer(
                    columns="a",
                    new_column_names={"a": "a2"},
                ),
            ),
            ("impute", MeanImputer(columns=["a2"])),
        ]

        lineage = build_lineage(steps)

        assert lineage["impute"].input_root_fields == {"a2": ["a"]}
        assert lineage["impute"].producer_step == {"a2": ["rename"]}

    def test_explicit_root_columns(self):
        steps = [("impute", MeanImputer(columns=["a"]))]

        lineage = build_lineage(steps, root_columns=["a", "unused_root"])

        assert lineage["impute"].input_root_fields == {"a": ["a"]}

    def test_indirect_reference_resolved_through_lineage(self):
        steps = [
            (
                "rename",
                RenameColumnsTransformer(
                    columns="w",
                    new_column_names={"w": "w2"},
                ),
            ),
            ("impute", MeanImputer(columns=["a"], weights_column="w2")),
        ]

        lineage = build_lineage(steps)

        assert lineage["impute"].input_root_fields == {
            "a": ["a"],
            "w2": ["w"],
        }

    def test_fits_flag_recorded_per_step(self):
        steps = [
            ("cast", ColumnDtypeSetter(columns="a", dtype="Float64")),
            ("impute", MeanImputer(columns=["a"])),
        ]

        lineage = build_lineage(steps)

        assert lineage["cast"].fits is False
        assert lineage["impute"].fits is True

    def test_to_dict_is_json_compatible(self):
        steps = [("impute", MeanImputer(columns=["a"]))]

        record = build_lineage(steps)["impute"]

        serialized = record.to_dict()

        assert isinstance(serialized, dict)
        assert serialized["step_name"] == "impute"
        assert isinstance(serialized["input_root_fields"], dict)


class TestAnalyzeImpact:
    def _lineage(self):
        return build_lineage(
            [
                ("cast", ColumnDtypeSetter(columns="a", dtype="Float64")),
                ("diff", DifferenceTransformer(columns=["a", "b"])),
                (
                    "kmeans",
                    OneDKmeansTransformer(
                        columns="a_minus_b",
                        new_column_name="cluster",
                        n_clusters=2,
                    ),
                ),
                ("impute_c", MeanImputer(columns=["c"])),
            ],
        )

    def test_change_propagates_to_dependent_fitted_step(self):
        impact = analyze_impact(self._lineage(), ["diff"])

        assert impact["affected_steps"] == ["diff", "kmeans"]
        assert impact["refit_steps"] == ["kmeans"]
        assert impact["reuse_steps"] == ["cast", "impute_c"]
        assert impact["affected_columns"]["kmeans"] == ["a_minus_b"]

    def test_independent_step_is_reused(self):
        impact = analyze_impact(self._lineage(), ["diff"])

        assert "impute_c" in impact["reuse_steps"]
        assert "impute_c" not in impact["affected_steps"]

    def test_change_to_root_step_propagates_everywhere_dependent(self):
        impact = analyze_impact(self._lineage(), ["cast"])

        assert impact["refit_steps"] == ["kmeans"]
        # impute_c reads only root field c, which cast never touches
        assert impact["reuse_steps"] == ["impute_c"]

    def test_stateless_affected_step_not_in_refit(self):
        impact = analyze_impact(self._lineage(), ["cast"])

        assert "diff" in impact["affected_steps"]
        assert "diff" not in impact["refit_steps"]

    def test_unknown_changed_step_raises(self):
        with pytest.raises(KeyError, match="unknown step"):
            analyze_impact(self._lineage(), ["does_not_exist"])

    def test_impact_record_types(self):
        impact = analyze_impact(self._lineage(), ["diff"])

        assert isinstance(impact["changed_steps"], list)
        assert isinstance(impact["affected_columns"], dict)
