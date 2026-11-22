"""Methods for serializing, versioning and selectively refitting pipelines."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any, NamedTuple

from sklearn.pipeline import Pipeline

from tubular.base import CLASS_REGISTRY, BaseTransformer
from tubular.lineage import (
    StepLineage,
    analyze_impact,
    build_lineage,
    get_input_columns,
    step_requires_fit,
)

if TYPE_CHECKING:
    from tubular.types import DataFrame, Series

# Top-level keys holding pipeline metadata rather than transformer step json.
_METADATA_KEY = "_pipeline_metadata"

# A changed step is supplied either as a transformer instance or a step json.
ChangedSteps = dict[str, BaseTransformer | dict[str, Any]]


class RefitPlan(NamedTuple):
    """Step names produced by a selective refit run.

    Attributes
    ----------
    rebuilt : list[str]
        Every changed or affected step that was rebuilt.
    refitted : list[str]
        Subset of rebuilt steps that actually ran a fit.
    reused : list[str]
        Steps whose fitted state was restored unchanged from json.

    """

    rebuilt: list[str]
    refitted: list[str]
    reused: list[str]


def dump_pipeline_to_json(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    """Serialize a pipeline into json dictionary.

    Parameters
    ----------
    pipeline: Pipeline
        sequence of transformer objects

    Returns
    -------
    dict
        json dictionary representing the pipeline.

    Raises
    ------
    RuntimeError
        If any of the transformer in pipeline is not jsonable it raises RuntimeError.

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> from tubular.imputers import MeanImputer, MedianImputer
    >>> from sklearn.pipeline import Pipeline

    >>> df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
    >>> median_imputer = MedianImputer(columns=["b"])
    >>> mean_imputer = MeanImputer(columns=["b"])
    >>> original_pipeline = Pipeline(
    ...     [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
    ... )
    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline_json  # doctest: +NORMALIZE_WHITESPACE
    {'median_imputer': {'tubular_version':...,
    'classname': 'MedianImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
    'fit': {'is_fitted_': True, 'impute_values_': {'b': 15.0}}},
    'mean_imputer': {'tubular_version':...,
    'classname': 'MeanImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
        'fit': {'is_fitted_': True, 'impute_values_': {'b': 15.0}}}}

    ```

    """
    steps = pipeline.steps
    non_jsonable_steps = [step[0] for step in steps if step[1].jsonable is False]
    if non_jsonable_steps:
        msg = f"the following steps are not yet jsonable: {non_jsonable_steps}"
        raise RuntimeError(msg)

    return {step_name: step.to_json() for step_name, step in steps}


def load_pipeline_from_json(pipeline_json: dict[str, dict[str, Any]]) -> Pipeline:
    """Deserialize a pipeline json structure into a pipeline.

    Parameters
    ----------
    pipeline_json: dict
        json dictionary representing the pipeline.

    Returns
    -------
    Pipeline loaded  from json dict

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> from tubular.imputers import MeanImputer, MedianImputer
    >>> from sklearn.pipeline import Pipeline
    >>> df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
    >>> median_imputer = MedianImputer(columns=["b"])
    >>> mean_imputer = MeanImputer(columns=["b"])
    >>> original_pipeline = Pipeline(
    ...     [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
    ... )

    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline = load_pipeline_from_json(pipeline_json)
    >>> pipeline
    Pipeline(steps=[('median_imputer', MedianImputer(columns=['b'])),
                    ('mean_imputer', MeanImputer(columns=['b']))])

    ```

    """
    steps = [
        (step_name, CLASS_REGISTRY[json_dict["classname"]].from_json(json_dict))
        for step_name, json_dict in pipeline_json.items()
        if step_name != _METADATA_KEY
    ]

    return Pipeline(steps)


def _step_content_hash(step_json: dict[str, Any]) -> str:
    """Compute a stable hash of a step's json.

    Returns
    -------
    str
        12-character content hash used as the step's traceable version id.

    """
    payload = json.dumps(step_json, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _strip_versioned_keys(step_json: dict[str, Any]) -> dict[str, Any]:
    """Return the plain transformer json of a versioned step.

    Returns
    -------
    dict[str, Any]
        Step json without the ``version`` and ``lineage`` annotation keys.

    """
    return {
        key: value
        for key, value in step_json.items()
        if key not in {"version", "lineage"}
    }


def dump_versioned_pipeline_to_json(
    pipeline: Pipeline,
    root_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Serialize a pipeline with per-step lineage and traceable versions.

    The transformer json of each step is unchanged (so each step remains
    loadable with :func:`load_pipeline_from_json`); version and lineage are
    attached alongside it. A pipeline metadata block records step order,
    content versions and column lineage.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline to serialize.
    root_columns : list[str] or None, default = None
        Columns of the original training table. If ``None`` they are inferred
        from columns that have no producing step.

    Returns
    -------
    dict
        Versioned json dictionary. Each step gains ``version`` and
        ``lineage`` keys, and a ``_pipeline_metadata`` block is added.

    Raises
    ------
    RuntimeError
        If any pipeline step is not jsonable.

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> from sklearn.pipeline import Pipeline
    >>> from tubular.imputers import MeanImputer
    >>> from tubular.pipeline import dump_versioned_pipeline_to_json
    >>> df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
    >>> pipeline = Pipeline([("mean_imputer", MeanImputer(columns=["b"]))]).fit(df)
    >>> versioned = dump_versioned_pipeline_to_json(pipeline)
    >>> versioned["mean_imputer"]["version"]  # doctest: +SKIP
    'a1b2c3d4e5f6'
    >>> versioned["_pipeline_metadata"]["step_order"]
    ['mean_imputer']

    ```

    """
    steps = pipeline.steps
    non_jsonable_steps = [step[0] for step in steps if step[1].jsonable is False]
    if non_jsonable_steps:
        msg = f"the following steps are not yet jsonable: {non_jsonable_steps}"
        raise RuntimeError(msg)

    lineage = build_lineage(steps, root_columns=root_columns)

    versioned: dict[str, Any] = {}
    step_versions: dict[str, str] = {}

    for step_name, transformer in steps:
        step_json = transformer.to_json()
        version = _step_content_hash(step_json)
        step_versions[step_name] = version

        versioned[step_name] = {
            **step_json,
            "version": version,
            "lineage": lineage[step_name].to_dict(),
        }

    versioned[_METADATA_KEY] = {
        "step_order": [name for name, _ in steps],
        "step_versions": step_versions,
        "lineage": {name: record.to_dict() for name, record in lineage.items()},
        "version_history": {},
    }

    return versioned


def _lineage_from_versioned_json(
    pipeline_json: dict[str, Any],
) -> dict[str, StepLineage]:
    """Rebuild StepLineage objects from a versioned pipeline json.

    Returns
    -------
    dict[str, StepLineage]
        Lineage records keyed by step name.

    """
    metadata = pipeline_json.get(_METADATA_KEY, {})
    raw_lineage = metadata.get("lineage", {})

    lineage: dict[str, StepLineage] = {}
    for step_name, record in raw_lineage.items():
        lineage[step_name] = StepLineage(
            step_name=record["step_name"],
            classname=record["classname"],
            input_columns=record["input_columns"],
            output_columns=record["output_columns"],
            input_root_fields=record.get("input_root_fields", {}),
            output_root_fields=record.get("output_root_fields", {}),
            producer_step=record.get("producer_step", {}),
            fits=record.get("fits", True),
        )

    return lineage


def analyze_pipeline_impact(
    pipeline_json: dict[str, Any],
    changed_steps: list[str],
) -> dict[str, Any]:
    """Identify which downstream fitted steps must be refit.

    Uses the real-field lineage stored in a versioned pipeline json (see
    :func:`dump_versioned_pipeline_to_json`) to propagate the change through
    indirect column references.

    Parameters
    ----------
    pipeline_json : dict
        Versioned pipeline json.
    changed_steps : list[str]
        Steps whose transformer was updated.

    Returns
    -------
    dict
        Impact report with ``refit_steps`` (refit on the historical training
        table), ``reuse_steps`` (restore unchanged fitted state from json) and
        the affected columns per step.

    """
    lineage = _lineage_from_versioned_json(pipeline_json)
    return analyze_impact(lineage, changed_steps)


def _new_transformer_from_init(step_json: dict[str, Any]) -> BaseTransformer:
    """Build an unfitted transformer from the init section of a step json.

    Returns
    -------
    BaseTransformer
        New, unfitted transformer instance.

    """
    cls = CLASS_REGISTRY[step_json["classname"]]
    return cls(**step_json["init"])


def _validate_column_availability(
    steps: list[tuple[str, Any]],
    X: DataFrame,
    changed_steps: dict[str, Any],
) -> None:
    """Check every step can still see its referenced columns after the change.

    Columns are tracked statically through the lineage outputs rather than
    executing the pipeline, so a stale downstream reference to a renamed/
    removed column raises an actionable error before any fitting starts.

    Raises
    ------
    ValueError
        If a step (other than a user-provided changed step) references a
        column that is neither in the training table nor produced upstream.

    """
    available = (
        set(X.collect_schema().names())
        if hasattr(
            X,
            "collect_schema",
        )
        else set(X.columns)
    )

    for step_name, transformer in steps:
        missing = sorted(
            set(get_input_columns(transformer)).difference(available),
        )
        if missing:
            if step_name in changed_steps:
                msg = (
                    f"changed step {step_name} references column(s) {missing} "
                    "that are not available at that point in the pipeline"
                )
            else:
                msg = (
                    f"downstream step {step_name} still references column(s) "
                    f"{missing} removed or renamed by the updated step(s) "
                    f"{sorted(changed_steps)}; supply an updated transformer "
                    "for this step in changed_steps"
                )
            raise ValueError(msg)

        available.update(transformer.get_feature_names_out())


def _materialize_transformers(
    pipeline_json: dict[str, Any],
    changed_steps: ChangedSteps,
    step_order: list[str],
) -> tuple[dict[str, BaseTransformer], dict[str, dict[str, Any]]]:
    """Build the current transformer instance for every pipeline step.

    Changed steps use the supplied transformer or step json; all other steps
    are reconstructed unchanged from the original pipeline json.

    Returns
    -------
    tuple[dict, dict]
        Transformers keyed by step name, and plain json for any changed step
        supplied as a json dict.

    Raises
    ------
    ValueError
        If a supplied replacement transformer is not jsonable.

    """
    current_transformers: dict[str, BaseTransformer] = {}
    updated_step_json: dict[str, dict[str, Any]] = {}

    for step_name in step_order:
        if step_name in changed_steps:
            replacement = changed_steps[step_name]
            if isinstance(replacement, dict):
                transformer = CLASS_REGISTRY[replacement["classname"]].from_json(
                    replacement,
                )
                updated_step_json[step_name] = copy.deepcopy(replacement)
            else:
                transformer = replacement
                if getattr(transformer, "jsonable", True) is False:
                    msg = f"changed step {step_name} is not jsonable"
                    raise ValueError(msg)
        else:
            transformer = CLASS_REGISTRY[
                pipeline_json[step_name]["classname"]
            ].from_json(pipeline_json[step_name])
        current_transformers[step_name] = transformer

    return current_transformers, updated_step_json


def _run_selective_refit(  # noqa: PLR0913, PLR0917 - internal step-run orchestrator
    step_order: list[str],
    current_transformers: dict[str, BaseTransformer],
    updated_step_json: dict[str, dict[str, Any]],
    pipeline_json: dict[str, Any],
    refit_set: set[str],
    X: DataFrame,
    y: Series | None,
) -> tuple[dict[str, BaseTransformer], RefitPlan]:
    """Run the pipeline, fitting affected steps and reusing the rest.

    Returns
    -------
    tuple[dict, RefitPlan]
        Fitted transformers keyed by step name and the step lists from the
        refit run.

    """
    frame = X
    fitted_transformers: dict[str, BaseTransformer] = {}
    rebuilt: list[str] = []
    refitted: list[str] = []
    reused: list[str] = []

    for step_name in step_order:
        transformer = current_transformers[step_name]

        if step_name in refit_set:
            # json-reconstructed transformers cannot fit, so start from a
            # fresh instance built out of the init parameters.
            if getattr(transformer, "built_from_json", False):
                source_json = updated_step_json.get(
                    step_name,
                    pipeline_json[step_name],
                )
                transformer = _new_transformer_from_init(source_json)
            if step_requires_fit(transformer):
                transformer.fit(frame, y)
                refitted.append(step_name)
            rebuilt.append(step_name)
        else:
            reused.append(step_name)

        fitted_transformers[step_name] = transformer
        frame = transformer.transform(frame)

    return fitted_transformers, RefitPlan(rebuilt, refitted, reused)


def _build_refit_json(  # noqa: PLR0913, PLR0917 - internal json assembly helper
    pipeline_json: dict[str, Any],
    step_order: list[str],
    fitted_transformers: dict[str, BaseTransformer],
    plan: RefitPlan,
    changed_steps: ChangedSteps,
    impact: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the updated versioned json after a selective refit.

    Returns
    -------
    dict[str, Any]
        New versioned pipeline json with refreshed versions, lineage and
        per-step version history.

    """
    metadata = pipeline_json[_METADATA_KEY]
    new_pipeline = Pipeline(
        [(name, fitted_transformers[name]) for name in step_order],
    )
    new_lineage = build_lineage(new_pipeline.steps)

    new_step_json: dict[str, Any] = {}
    new_versions: dict[str, str] = {}
    rebuilt_set = set(plan.rebuilt)
    for step_name in step_order:
        if step_name in rebuilt_set:
            step_json = fitted_transformers[step_name].to_json()
        else:
            step_json = _strip_versioned_keys(pipeline_json[step_name])
        version = _step_content_hash(step_json)
        new_versions[step_name] = version
        new_step_json[step_name] = {
            **step_json,
            "version": version,
            "lineage": new_lineage[step_name].to_dict(),
        }

    new_metadata = {
        "step_order": step_order,
        "step_versions": new_versions,
        "lineage": {name: record.to_dict() for name, record in new_lineage.items()},
        "version_history": copy.deepcopy(metadata.get("version_history", {})),
    }

    for step_name in plan.rebuilt:
        old_version = metadata["step_versions"].get(step_name)
        new_version = new_versions[step_name]
        if old_version is None or old_version != new_version:
            reasons = sorted(changed_steps)
            entry = {
                "version": new_version,
                "previous_version": old_version,
                "reason": f"refit_after_change:{','.join(reasons)}",
            }
            new_metadata["version_history"].setdefault(step_name, []).append(entry)

    new_metadata["last_refit"] = {
        "changed_steps": sorted(changed_steps),
        "rebuilt_steps": plan.rebuilt,
        "refit_steps": plan.refitted,
        "reuse_steps": plan.reused,
        "impact": impact,
    }
    new_step_json[_METADATA_KEY] = new_metadata

    return new_step_json


def refit_pipeline(
    pipeline_json: dict[str, Any],
    changed_steps: ChangedSteps,
    X: DataFrame,
    y: Series | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Refit selectively the downstream steps affected by an upstream change.

    Steps in ``changed_steps`` replace the corresponding json step. Lineage
    analysis then determines every downstream step whose input columns derive
    from the changed outputs:

    * affected steps with genuine fitted state are refit on the historical
      training table ``X``;
    * stateless steps are rebuilt from their init parameters and re-run as
      part of the sequential transform;
    * unaffected steps are reconstructed unchanged from the original json,
      preserving their historical fitted state.

    Each refit step keeps a traceable version entry; the previous json version
    is retained under ``version_history`` rather than overwritten.

    Parameters
    ----------
    pipeline_json : dict
        Versioned pipeline json produced by
        :func:`dump_versioned_pipeline_to_json`.
    changed_steps : dict[str, BaseTransformer | dict]
        Mapping of step name to either a fitted/unfitted transformer instance
        or a full step json dict, for the steps that were updated.
    X : DataFrame
        Historical training table used to refit the affected downstream steps.
    y : Series or None, default = None
        Optional response passed to ``fit`` where required.

    Returns
    -------
    tuple[Pipeline, dict]
        The refitted pipeline and an updated versioned pipeline json (with
        lineage, versions and version history refreshed).

    Raises
    ------
    KeyError
        If a changed step name is not in the pipeline json.
    ValueError
        If ``pipeline_json`` is not a versioned pipeline json, a replacement
        is not jsonable, or a downstream step references a column removed or
        renamed by the updated step(s).

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> from sklearn.pipeline import Pipeline
    >>> from tubular.imputers import MeanImputer
    >>> from tubular.numeric import DifferenceTransformer, OneDKmeansTransformer
    >>> from tubular.pipeline import (
    ...     dump_versioned_pipeline_to_json,
    ...     refit_pipeline,
    ... )
    >>> df = pl.DataFrame(
    ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0],
    ...      "c": [10.0, 20.0, 30.0, 40.0]}
    ... )
    >>> pipeline = Pipeline(
    ...     [
    ...         ("diff", DifferenceTransformer(columns=["a", "b"])),
    ...         ("kmeans", OneDKmeansTransformer(
    ...             columns="a_minus_b", new_column_name="cluster", n_clusters=2
    ...         )),
    ...         ("impute_c", MeanImputer(columns=["c"])),
    ...     ]
    ... ).fit(df)
    >>> pipeline_json = dump_versioned_pipeline_to_json(pipeline)
    >>> new_steps = {
    ...     "diff": DifferenceTransformer(columns=["a", "c"]),
    ...     "kmeans": OneDKmeansTransformer(
    ...         columns="a_minus_c", new_column_name="cluster", n_clusters=2
    ...     ),
    ... }
    >>> refitted, new_json = refit_pipeline(pipeline_json, new_steps, df)
    >>> new_json["_pipeline_metadata"]["version_history"]["kmeans"]  # doctest: +SKIP
    [{'version': '...', 'reason': 'refit_after_change:diff,kmeans'}]

    ```

    """
    metadata = pipeline_json.get(_METADATA_KEY)
    if metadata is None:
        msg = (
            "pipeline_json must be a versioned pipeline json from "
            "dump_versioned_pipeline_to_json"
        )
        raise ValueError(msg)

    step_order = list(metadata["step_order"])
    unknown = [name for name in changed_steps if name not in pipeline_json]
    if unknown:
        msg = f"changed step(s) not present in pipeline json: {unknown}"
        raise KeyError(msg)

    current_transformers, updated_step_json = _materialize_transformers(
        pipeline_json,
        changed_steps,
        step_order,
    )

    steps = [(name, current_transformers[name]) for name in step_order]
    lineage = build_lineage(steps)
    impact = analyze_impact(lineage, list(changed_steps))
    refit_set = set(impact["refit_steps"]) | set(changed_steps)

    _validate_column_availability(steps, X, changed_steps)

    fitted_transformers, plan = _run_selective_refit(
        step_order,
        current_transformers,
        updated_step_json,
        pipeline_json,
        refit_set,
        X,
        y,
    )

    new_pipeline = Pipeline(
        [(name, fitted_transformers[name]) for name in step_order],
    )

    new_json = _build_refit_json(
        pipeline_json,
        step_order,
        fitted_transformers,
        plan,
        changed_steps,
        impact,
    )

    return new_pipeline, new_json
