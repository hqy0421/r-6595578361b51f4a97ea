"""Column lineage and refit impact analysis for pipelines.

When an intermediate step of a fitted pipeline is updated, the meaning of the
columns seen by downstream steps can change. This module resolves, for every
column produced by a pipeline, which *real* (root) columns of the original
training table it can be traced back to, through both the direct
``columns`` argument and the indirect column references held in other
transformer attributes (e.g. ``weights_column``, ``when_column``, ``key``).

The lineage is then used to distinguish the downstream fitted transformers
that must be refit on the historical training table from those whose fitted
state is unaffected and can be restored from the original json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tubular.base import BaseTransformer

# Transformer attributes (other than ``columns``) that hold the name of a
# column read at fit/transform time. They are only treated as references when
# the attribute is present, holds a string, and is not gated off by a
# companion flag (see ``_CONDITIONAL_REFERENCE_ATTRS``).
_INDIRECT_REFERENCE_ATTRS: tuple[str, ...] = (
    "weights_column",
    "when_column",
    "then_column",
    "key",
)

# Conditional references: attribute only names a column when a companion flag
# attribute is truthy. e.g. StringContainsTransformer(reference=...,
# reference_as_column=True).
_CONDITIONAL_REFERENCE_ATTRS: tuple[tuple[str, str], ...] = (
    ("reference", "reference_as_column"),
)


def get_input_columns(transformer: BaseTransformer) -> list[str]:
    """All real columns a transformer reads from the frame.

    This combines the direct ``columns`` argument with any indirect column
    references held in other attributes, so impact analysis is not limited to
    the ``columns`` list.

    Parameters
    ----------
    transformer : BaseTransformer
        Transformer to inspect.

    Returns
    -------
    list[str]
        Sorted, de-duplicated list of input column names referenced by the
        transformer.

    Examples
    --------
    ```pycon
    >>> from tubular.imputers import MeanImputer
    >>> from tubular.lineage import get_input_columns
    >>> transformer = MeanImputer(columns=["a", "b"], weights_column="w")
    >>> get_input_columns(transformer)
    ['a', 'b', 'w']

    ```

    """
    referenced: set[str] = set()

    referenced.update(getattr(transformer, "columns", []) or [])

    for attr in _INDIRECT_REFERENCE_ATTRS:
        value = getattr(transformer, attr, None)
        if isinstance(value, str):
            referenced.add(value)

    for ref_attr, flag_attr in _CONDITIONAL_REFERENCE_ATTRS:
        if getattr(transformer, flag_attr, False):
            value = getattr(transformer, ref_attr, None)
            if isinstance(value, str):
                referenced.add(value)

    return sorted(referenced)


def get_output_columns(transformer: BaseTransformer) -> list[str]:
    """Columns a transformer modifies or creates.

    Parameters
    ----------
    transformer : BaseTransformer
        Transformer to inspect.

    Returns
    -------
    list[str]
        Output feature names as reported by ``get_feature_names_out``.

    """
    return list(transformer.get_feature_names_out())


@dataclass
class StepLineage:
    """Lineage information for a single pipeline step.

    Attributes
    ----------
    step_name : str
        Name of the pipeline step.
    classname : str
        Class name of the transformer.
    input_columns : list[str]
        Real columns the step reads, including indirect references.
    output_columns : list[str]
        Columns the step modifies or creates.
    input_root_fields : dict[str, list[str]]
        Mapping from each input column to the root training-table fields it
        traces back to. An input column that is itself a root field maps to
        itself.
    output_root_fields : dict[str, list[str]]
        Mapping from each output column to the root fields it traces back to.
    producer_step : dict[str, list[str]]
        Mapping from each input column to the name of the upstream step that
        (last) produced it, wrapped in a list. Root fields are not present as
        keys.
    fits : bool
        Whether the step carries learned state that requires refitting when an
        upstream meaning changes.

    """

    step_name: str
    classname: str
    input_columns: list[str]
    output_columns: list[str]
    input_root_fields: dict[str, list[str]] = field(default_factory=dict)
    output_root_fields: dict[str, list[str]] = field(default_factory=dict)
    producer_step: dict[str, list[str]] = field(default_factory=dict)
    fits: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a json-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Plain dictionary representation of the lineage record.

        """
        return {
            "step_name": self.step_name,
            "classname": self.classname,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "input_root_fields": self.input_root_fields,
            "output_root_fields": self.output_root_fields,
            "producer_step": self.producer_step,
            "fits": self.fits,
        }


def build_lineage(
    steps: list[tuple[str, BaseTransformer]],
    root_columns: list[str] | None = None,
) -> dict[str, StepLineage]:
    """Build per-step column lineage through a pipeline.

    Each live column is tracked with the set of root (original training table)
    fields it ultimately derives from and the step that last produced it.
    Indirect references (e.g. ``weights_column``) are resolved through the same
    lineage map as direct ``columns`` references.

    Parameters
    ----------
    steps : list[tuple[str, BaseTransformer]]
        Ordered pipeline steps as ``(step_name, transformer)`` pairs.
    root_columns : list[str] or None, default = None
        Columns present in the original training table. If ``None``, every
        column referenced by the first step that has no known producer is
        treated as a root field.

    Returns
    -------
    dict[str, StepLineage]
        Lineage record keyed by step name.

    Examples
    --------
    ```pycon
    >>> import polars as pl
    >>> from sklearn.pipeline import Pipeline
    >>> from tubular.numeric import DifferenceTransformer, OneDKmeansTransformer
    >>> from tubular.lineage import build_lineage
    >>> pipeline = Pipeline(
    ...     [
    ...         ("diff", DifferenceTransformer(columns=["a", "b"])),
    ...         ("kmeans", OneDKmeansTransformer(
    ...             columns="a_minus_b", new_column_name="cluster", n_clusters=2
    ...         )),
    ...     ]
    ... )
    >>> lineage = build_lineage(pipeline.steps)
    >>> lineage["kmeans"].input_root_fields
    {'a_minus_b': ['a', 'b']}

    ```

    """
    root_set = set(root_columns) if root_columns is not None else set()

    # column -> root fields it derives from
    column_roots: dict[str, set[str]] = {col: {col} for col in root_set}
    # column -> step that last produced/modified it
    column_producer: dict[str, str] = {}

    lineage: dict[str, StepLineage] = {}

    for step_name, transformer in steps:
        input_columns = get_input_columns(transformer)
        output_columns = get_output_columns(transformer)

        # Any referenced column with no known producer is an original field.
        for column in input_columns:
            if column not in column_roots:
                column_roots[column] = {column}
                root_set.add(column)

        input_root_fields = {
            column: sorted(column_roots.get(column, {column}))
            for column in input_columns
        }
        producer_step = {
            column: [column_producer[column]]
            for column in input_columns
            if column in column_producer
        }

        # An output inherits the union of the root fields of every input.
        combined_roots: set[str] = set()
        for column in input_columns:
            combined_roots.update(column_roots.get(column, {column}))

        output_root_fields = {
            column: sorted(combined_roots) for column in output_columns
        }

        record = StepLineage(
            step_name=step_name,
            classname=type(transformer).__name__,
            input_columns=input_columns,
            output_columns=output_columns,
            input_root_fields=input_root_fields,
            output_root_fields=output_root_fields,
            producer_step=producer_step,
            fits=step_requires_fit(transformer),
        )
        lineage[step_name] = record

        # Outputs become the current version of those columns; if an output
        # name is new it inherits the combined roots, otherwise it overwrites
        # the column's prior lineage.
        for column in output_columns:
            column_roots[column] = set(combined_roots)
            column_producer[column] = step_name

    return lineage


def step_requires_fit(transformer: BaseTransformer) -> bool:
    """Whether a transformer holds learned state that can become stale.

    Stateless transformers declare ``FITS = False`` or set
    ``is_fitted_ = True`` inside ``__init__``; they are deterministic column
    expressions and only need to be re-run (``transform``), never refit.

    Parameters
    ----------
    transformer : BaseTransformer
        Transformer to inspect.

    Returns
    -------
    bool
        ``True`` if the transformer has a genuine fit stage.

    Examples
    --------
    ```pycon
    >>> from tubular.imputers import MeanImputer
    >>> from tubular.numeric import DifferenceTransformer
    >>> from tubular.lineage import step_requires_fit
    >>> step_requires_fit(MeanImputer(columns="a"))
    True
    >>> step_requires_fit(DifferenceTransformer(columns=["a", "b"]))
    False

    ```

    """
    if not getattr(transformer, "FITS", True):
        return False

    # Stateless transformers hard-code is_fitted_=True in __init__.
    return "is_fitted_" not in type(transformer).__init__.__code__.co_consts


def analyze_impact(
    lineage: dict[str, StepLineage],
    changed_steps: list[str],
) -> dict[str, Any]:
    """Propagate a change to upstream steps through the column lineage.

    A downstream step is affected when any column it references (directly or
    indirectly) was produced by one of the changed steps, or derives from a
    column produced by an already-affected step. Among affected steps, only
    those with genuine fitted state need refitting; stateless steps simply
    propagate the changed columns.

    Parameters
    ----------
    lineage : dict[str, StepLineage]
        Lineage returned by :func:`build_lineage`.
    changed_steps : list[str]
        Names of the steps that were updated.

    Returns
    -------
    dict
        ``affected_steps``: every step downstream whose input meaning changed,
        in pipeline order. ``refit_steps``: subset that must be refit on the
        historical training table. ``reuse_steps``: steps whose fitted state is
        unaffected and can be restored from the original json.
        ``affected_columns``: columns whose meaning changed, keyed by step.

    Raises
    ------
    KeyError
        If a named changed step is not present in the lineage.

    Examples
    --------
    ```pycon
    >>> from tubular.imputers import MeanImputer
    >>> from tubular.numeric import DifferenceTransformer, OneDKmeansTransformer
    >>> from tubular.lineage import analyze_impact, build_lineage
    >>> steps = [
    ...     ("diff", DifferenceTransformer(columns=["a", "b"])),
    ...     ("kmeans", OneDKmeansTransformer(
    ...         columns="a_minus_b", new_column_name="cluster", n_clusters=2
    ...     )),
    ...     ("impute_c", MeanImputer(columns=["c"])),
    ... ]
    >>> lineage = build_lineage(steps)
    >>> impact = analyze_impact(lineage, changed_steps=["diff"])
    >>> impact["refit_steps"]
    ['kmeans']
    >>> impact["reuse_steps"]
    ['impute_c']

    ```

    """
    unknown = [name for name in changed_steps if name not in lineage]
    if unknown:
        msg = f"unknown step(s) in changed_steps: {unknown}"
        raise KeyError(msg)

    affected: set[str] = set(changed_steps)
    # columns whose meaning is currently "dirty", keyed by producing step
    dirty_columns: set[str] = set()
    for name in changed_steps:
        dirty_columns.update(lineage[name].output_columns)

    affected_columns: dict[str, list[str]] = {
        name: sorted(lineage[name].output_columns) for name in changed_steps
    }

    ordered_names = list(lineage)
    for step_name in ordered_names:
        if step_name in affected:
            continue

        record = lineage[step_name]
        hit_columns = sorted(
            set(record.input_columns).intersection(dirty_columns),
        )
        if not hit_columns:
            continue

        affected.add(step_name)
        affected_columns[step_name] = hit_columns
        dirty_columns.update(record.output_columns)

    refit_steps = [
        name
        for name in ordered_names
        if name in affected and lineage[name].fits and name not in changed_steps
    ]
    reuse_steps = [
        name
        for name in ordered_names
        if name not in affected and name not in changed_steps
    ]

    return {
        "changed_steps": list(changed_steps),
        "affected_steps": [name for name in ordered_names if name in affected],
        "refit_steps": refit_steps,
        "reuse_steps": reuse_steps,
        "affected_columns": affected_columns,
    }
