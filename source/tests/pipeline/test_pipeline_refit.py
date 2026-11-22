"""Tests for versioned pipeline json, impact analysis and selective refit."""

import copy

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from tubular.imputers import MeanImputer
from tubular.misc import ColumnDtypeSetter, RenameColumnsTransformer
from tubular.numeric import DifferenceTransformer, OneDKmeansTransformer
from tubular.pipeline import (
    _METADATA_KEY,
    analyze_pipeline_impact,
    dump_pipeline_to_json,
    dump_versioned_pipeline_to_json,
    load_pipeline_from_json,
    refit_pipeline,
)

VERSION_HASH_LENGTH = 12
EXPECTED_HISTORY_LENGTH_AFTER_TWO_REFITS = 2


@pytest.fixture
def training_df():
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "d": [1.0, None, 3.0, None, 5.0, 6.0],
        },
    )


@pytest.fixture
def fitted_pipeline(training_df):
    pipeline = Pipeline(
        [
            ("cast_a", ColumnDtypeSetter(columns="a", dtype="Float64")),
            ("diff", DifferenceTransformer(columns=["a", "b"])),
            (
                "kmeans",
                OneDKmeansTransformer(
                    columns="a_minus_b",
                    new_column_name="cluster",
                    n_clusters=2,
                    kmeans_kwargs={"random_state": 42},
                ),
            ),
            ("impute_d", MeanImputer(columns=["d"])),
        ],
    )
    return pipeline.fit(training_df)


@pytest.fixture
def versioned_json(fitted_pipeline):
    return dump_versioned_pipeline_to_json(fitted_pipeline)


class TestDumpVersionedPipelineToJson:
    def test_step_json_content_unchanged(self, fitted_pipeline, versioned_json):
        plain = dump_pipeline_to_json(fitted_pipeline)

        for step_name, step_json in plain.items():
            for key, value in step_json.items():
                assert versioned_json[step_name][key] == value

    def test_version_and_lineage_attached_per_step(self, versioned_json):
        for step_name in versioned_json[_METADATA_KEY]["step_order"]:
            assert isinstance(versioned_json[step_name]["version"], str)
            assert len(versioned_json[step_name]["version"]) == VERSION_HASH_LENGTH
            assert "input_root_fields" in versioned_json[step_name]["lineage"]

    def test_metadata_block(self, versioned_json):
        metadata = versioned_json[_METADATA_KEY]

        assert metadata["step_order"] == [
            "cast_a",
            "diff",
            "kmeans",
            "impute_d",
        ]
        assert set(metadata["step_versions"]) == set(metadata["step_order"])
        assert metadata["version_history"] == {}
        assert set(metadata["lineage"]) == set(metadata["step_order"])

    def test_versioned_json_loads_as_pipeline(self, versioned_json):
        pipeline = load_pipeline_from_json(versioned_json)

        assert [name for name, _ in pipeline.steps] == versioned_json[_METADATA_KEY][
            "step_order"
        ]

    def test_lineage_resolves_real_fields(self, versioned_json):
        assert versioned_json["kmeans"]["lineage"]["input_root_fields"] == {
            "a_minus_b": ["a", "b"],
        }
        assert versioned_json["impute_d"]["lineage"]["input_root_fields"] == {
            "d": ["d"],
        }


class TestAnalyzePipelineImpact:
    def test_refit_and_reuse_partition(self, versioned_json):
        impact = analyze_pipeline_impact(versioned_json, ["diff"])

        assert impact["refit_steps"] == ["kmeans"]
        assert impact["reuse_steps"] == ["cast_a", "impute_d"]

    def test_change_to_unrelated_step(self, versioned_json):
        impact = analyze_pipeline_impact(versioned_json, ["impute_d"])

        assert impact["refit_steps"] == []
        assert impact["reuse_steps"] == ["cast_a", "diff", "kmeans"]


class TestRefitPipeline:
    def _updated_steps(self):
        return {
            "diff": DifferenceTransformer(columns=["a", "c"]),
            "kmeans": OneDKmeansTransformer(
                columns="a_minus_c",
                new_column_name="cluster",
                n_clusters=2,
                kmeans_kwargs={"random_state": 42},
            ),
        }

    def test_unaffected_fitted_state_restored_from_json(
        self,
        versioned_json,
        fitted_pipeline,
        training_df,
    ):
        original_impute = copy.deepcopy(
            fitted_pipeline.named_steps["impute_d"].impute_values_,
        )

        refitted, new_json = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )

        assert refitted.named_steps["impute_d"].impute_values_ == original_impute
        assert new_json[_METADATA_KEY]["last_refit"]["reuse_steps"] == [
            "cast_a",
            "impute_d",
        ]

    def test_affected_fitted_step_refit_on_training_table(
        self,
        versioned_json,
        fitted_pipeline,
        training_df,
    ):
        original_bins = list(fitted_pipeline.named_steps["kmeans"].bins)

        refitted, _ = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )

        assert list(refitted.named_steps["kmeans"].bins) != original_bins
        output = refitted.transform(training_df)
        assert "a_minus_c" in output.columns
        assert "a_minus_b" not in output.columns

    def test_traceable_version_history_retained(
        self,
        versioned_json,
        training_df,
    ):
        old_versions = versioned_json[_METADATA_KEY]["step_versions"]

        _, new_json = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )

        history = new_json[_METADATA_KEY]["version_history"]
        assert set(history) == {"diff", "kmeans"}
        for step_name in ("diff", "kmeans"):
            entry = history[step_name][0]
            assert entry["previous_version"] == old_versions[step_name]
            assert (
                entry["version"] == new_json[_METADATA_KEY]["step_versions"][step_name]
            )
            assert "refit_after_change" in entry["reason"]
        # untouched steps have no history entries
        assert "impute_d" not in history
        assert "cast_a" not in history

    def test_reused_step_version_unchanged(self, versioned_json, training_df):
        _, new_json = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )

        old_versions = versioned_json[_METADATA_KEY]["step_versions"]
        new_versions = new_json[_METADATA_KEY]["step_versions"]
        assert new_versions["impute_d"] == old_versions["impute_d"]
        assert new_versions["cast_a"] == old_versions["cast_a"]

    def test_history_accumulates_across_refits(
        self,
        versioned_json,
        training_df,
    ):
        _, first_json = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )
        _, second_json = refit_pipeline(
            first_json,
            {
                "diff": DifferenceTransformer(columns=["b", "c"]),
                "kmeans": OneDKmeansTransformer(
                    columns="b_minus_c",
                    new_column_name="cluster",
                    n_clusters=2,
                    kmeans_kwargs={"random_state": 42},
                ),
            },
            training_df,
        )

        history = second_json[_METADATA_KEY]["version_history"]
        assert len(history["diff"]) == EXPECTED_HISTORY_LENGTH_AFTER_TWO_REFITS
        assert len(history["kmeans"]) == EXPECTED_HISTORY_LENGTH_AFTER_TWO_REFITS

    def test_stale_downstream_reference_raises_actionable_error(
        self,
        versioned_json,
        training_df,
    ):
        with pytest.raises(ValueError, match="still references column"):
            refit_pipeline(
                versioned_json,
                {"diff": DifferenceTransformer(columns=["a", "c"])},
                training_df,
            )

    def test_unknown_changed_step_raises(self, versioned_json, training_df):
        with pytest.raises(KeyError, match="not present"):
            refit_pipeline(
                versioned_json,
                {"missing_step": MeanImputer(columns=["a"])},
                training_df,
            )

    def test_refit_requires_versioned_json(self, fitted_pipeline, training_df):
        plain = dump_pipeline_to_json(fitted_pipeline)

        with pytest.raises(ValueError, match="versioned pipeline json"):
            refit_pipeline(plain, self._updated_steps(), training_df)

    def test_refit_with_step_json_dict_replacement(
        self,
        versioned_json,
        training_df,
    ):
        replacement_json = DifferenceTransformer(
            columns=["a", "c"],
        ).to_json()

        with pytest.raises(ValueError, match="still references column"):
            refit_pipeline(
                versioned_json,
                {"diff": replacement_json},
                training_df,
            )

    def test_indirect_reference_drives_impact(
        self,
        training_df,
    ):
        pipeline = Pipeline(
            [
                (
                    "rename",
                    RenameColumnsTransformer(
                        columns="a",
                        new_column_names={"a": "a2"},
                    ),
                ),
                ("impute_a2", MeanImputer(columns=["a2"])),
                ("impute_b", MeanImputer(columns=["b"])),
            ],
        ).fit(training_df)
        versioned = dump_versioned_pipeline_to_json(pipeline)

        impact = analyze_pipeline_impact(versioned, ["rename"])

        assert impact["refit_steps"] == ["impute_a2"]
        assert impact["reuse_steps"] == ["impute_b"]

    def test_refitted_pipeline_output_matches_manual_fit(
        self,
        versioned_json,
        training_df,
    ):
        refitted, _ = refit_pipeline(
            versioned_json,
            self._updated_steps(),
            training_df,
        )

        manual = Pipeline(
            [
                ("cast_a", ColumnDtypeSetter(columns="a", dtype="Float64")),
                ("diff", DifferenceTransformer(columns=["a", "c"])),
                (
                    "kmeans",
                    OneDKmeansTransformer(
                        columns="a_minus_c",
                        new_column_name="cluster",
                        n_clusters=2,
                        kmeans_kwargs={"random_state": 42},
                    ),
                ),
                ("impute_d", MeanImputer(columns=["d"])),
            ],
        ).fit(training_df)

        assert refitted.transform(training_df).equals(
            manual.transform(training_df),
        )
