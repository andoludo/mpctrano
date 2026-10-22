"""Calibration parameters declared in the Trano YAML of a building.

The YAML stays the single source of truth: a numeric field to calibrate is written as a mapping with its nominal
value and its bounds, instead of the bare number Trano expects::

    material:
      - id: ROCKWOOL:CAL
        thermal_conductivity: {value: 0.04, parameter: true, min: 0.025, max: 0.08}
    spaces:
      - id: SPACE:001
        parameters:
          floor_area: 100.0
          average_room_height: 2.5

:class:`CalibrationSpec` reads such a file: it collects the annotated fields by path (the path convention of
``trano.runtime``: ``material[ROCKWOOL:CAL].thermal_conductivity``, ``spaces[SPACE:001].parameters.floor_area``)
and writes the flat copy Trano reads (every annotated field at its value, or at the values given). The link between
a YAML field and the parameters of the CasADi model (a conductivity moves the conductances of every layer made of
that material, a thickness moves them and the capacities) is measured, not assumed: :meth:`CalibrationSpec.groups`
regenerates the model with each field perturbed and reads the exponent of every parameter that moved
(:class:`~mpctrano.calibration.YamlGroup`), which the calibration then adjusts together.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mpctrano.calibration import YamlGroup
from mpctrano.model import FAST_MODEL_LIBRARY, FastModel

INCLUDE_DEFAULT = "!include_default"
PERTURBATION = 0.2
"""Relative change of a field when its sensitivities are measured."""
EXPONENT_TOLERANCE = 1e-6


class IncludeDefault:
    """The ``!include_default`` tag of a Trano YAML, kept as it is when the file is rewritten."""

    def __repr__(self) -> str:
        return INCLUDE_DEFAULT


class _Loader(yaml.SafeLoader):
    pass


class _Dumper(yaml.SafeDumper):
    pass


_Loader.add_constructor(INCLUDE_DEFAULT, lambda loader, node: IncludeDefault())
_Dumper.add_representer(IncludeDefault, lambda dumper, data: dumper.represent_scalar(INCLUDE_DEFAULT, ""))


class YamlParameter(BaseModel):
    """A numeric field of the YAML declared as a calibration parameter."""

    model_config = ConfigDict(frozen=True)

    path: str
    value: float
    lower: float = Field(alias="min")
    upper: float = Field(alias="max")

    @model_validator(mode="after")
    def _ordered(self) -> YamlParameter:
        if not self.lower <= self.value <= self.upper:
            raise ValueError(f"{self.path}: the value {self.value} is outside [{self.lower}, {self.upper}].")
        if self.lower <= 0:
            raise ValueError(f"{self.path}: the bounds must be positive (the field is scaled).")
        return self

    @property
    def name(self) -> str:
        """The last key of the path (``thermal_conductivity``)."""
        return self.path.rsplit(".", 1)[-1]


def _is_annotation(node: Any) -> bool:  # noqa: ANN401
    return isinstance(node, dict) and node.get("parameter") is True and "value" in node


def _item_key(item: Any, index: int) -> str:  # noqa: ANN401
    """The selector of a list item: its ``id``, the ``id`` of its single child (``systems``), or its index."""
    if isinstance(item, dict):
        if "id" in item:
            return str(item["id"])
        if len(item) == 1:
            (child,) = item.values()
            if isinstance(child, dict) and "id" in child:
                return str(child["id"])
    return str(index)


def _walk(node: Any, path: str) -> Iterator[tuple[str, dict[str, Any]]]:  # noqa: ANN401
    if isinstance(node, dict):
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _is_annotation(child):
                yield child_path, child
            else:
                yield from _walk(child, child_path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk(item, f"{path}[{_item_key(item, index)}]")


def _flatten(node: Any, values: Mapping[str, float], path: str) -> Any:  # noqa: ANN401
    """The YAML with every annotation replaced by its value (``values`` override the nominal ones)."""
    if isinstance(node, dict):
        if _is_annotation(node):
            return float(values.get(path, node["value"]))
        return {key: _flatten(child, values, f"{path}.{key}" if path else str(key)) for key, child in node.items()}
    if isinstance(node, list):
        return [_flatten(item, values, f"{path}[{_item_key(item, index)}]") for index, item in enumerate(node)]
    return node


@dataclass
class CalibrationSpec:
    """The annotated YAML of a building: its parameters and the flat copies Trano reads."""

    source: Path
    description: dict[str, Any]
    parameters: list[YamlParameter]
    base_dir: Path | None = None
    """Where the weather file of the building is looked up (the directory of the YAML by default)."""
    _directory: Path | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_yaml(cls, path: Path | str, base_dir: Path | str | None = None) -> CalibrationSpec:
        source = Path(path).resolve()
        description = yaml.load(source.read_text(), Loader=_Loader)  # noqa: S506 - a SafeLoader with one tag
        if not isinstance(description, dict):
            raise ValueError(f"{source} does not hold a YAML mapping.")
        parameters = [
            YamlParameter(path=p, **{k: v for k, v in node.items() if k != "parameter"})
            for p, node in _walk(description, "")
        ]
        return cls(
            source=source,
            description=description,
            parameters=parameters,
            base_dir=Path(base_dir).resolve() if base_dir is not None else None,
        )

    @property
    def paths(self) -> list[str]:
        return [parameter.path for parameter in self.parameters]

    @property
    def nominal(self) -> dict[str, float]:
        """The nominal value of each parameter, by path."""
        return {parameter.path: parameter.value for parameter in self.parameters}

    def __getitem__(self, path: str) -> YamlParameter:
        for parameter in self.parameters:
            if parameter.path == path:
                return parameter
        raise KeyError(f"No calibration parameter {path!r}; declared: {self.paths}")

    def write(self, values: Mapping[str, float] | None = None, directory: Path | None = None) -> Path:
        """Write the flat YAML (the annotated fields at ``values``, nominal otherwise) and return its path.

        Without ``directory`` the file goes to a temporary directory of the spec; the weather file and the other
        relative paths stay those of the source (the model is generated with the source directory as base).
        """
        values = dict(values or {})
        unknown = set(values) - set(self.paths)
        if unknown:
            raise KeyError(f"Unknown calibration parameters: {sorted(unknown)}")
        if directory is None:
            if self._directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="mpctrano-"))
            directory = self._directory
        stem = self.source.stem
        suffix = "" if not values else "-" + "-".join(f"{k.rsplit('.', 1)[-1]}{v:g}" for k, v in sorted(values.items()))
        path = Path(directory) / f"{stem}{suffix}.yaml"
        path.write_text(yaml.dump(_flatten(self.description, values, ""), Dumper=_Dumper, sort_keys=False))
        return path

    def model(
        self, library: str = FAST_MODEL_LIBRARY, values: Mapping[str, float] | None = None, name: str | None = None
    ) -> FastModel:
        """The model of the building with the parameters at ``values`` (nominal otherwise)."""
        base_dir = self.base_dir if self.base_dir is not None else self.source.parent
        return FastModel.from_yaml(self.write(values), name=name, library=library, base_dir=base_dir)

    def sensitivities(
        self,
        library: str = FAST_MODEL_LIBRARY,
        model: FastModel | None = None,
        values: Mapping[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """For each parameter, the exponent of every parameter of the model it moves (``p = p0 * (field /
        field0) ** e``), measured by regenerating the model with the field raised by :data:`PERTURBATION` from
        ``values`` (the nominal ones by default; ``model``, when given, is the model at ``values``).

        A conductivity moves the conductances of its layers with the exponent 1, a thickness moves them with -1 and
        the capacities with 1; a dependence that is not a power law (an optical fit) is approximated by the
        exponent measured at the perturbation.
        """
        values = dict(values or {})
        base = model or self.model(library, values)
        names = base.parameter_names
        result: dict[str, dict[str, float]] = {}
        for parameter in self.parameters:
            at = values.get(parameter.path, parameter.value)
            raised = self.model(library, {**values, parameter.path: at * (1 + PERTURBATION)})
            if raised.parameter_names != names:
                raise ValueError(f"{parameter.path} changes the structure of the model, not only its parameters.")
            exponents: dict[str, float] = {}
            for i, name in enumerate(names):
                before, after = float(base.parameters[i]), float(raised.parameters[i])
                if not (np.isfinite(before) and np.isfinite(after)) or before in (0.0, after):
                    continue
                if after * before <= 0:
                    raise ValueError(f"{parameter.path} changes the sign of {name}: not a scale.")
                exponent = float(np.log(after / before) / np.log(1 + PERTURBATION))
                if abs(exponent) > EXPONENT_TOLERANCE:
                    exponents[name] = exponent
            if not exponents:
                raise ValueError(f"{parameter.path} moves no parameter of the model (the equations do not use it).")
            result[parameter.path] = exponents
        return result

    def groups(
        self,
        model: FastModel | None = None,
        library: str = FAST_MODEL_LIBRARY,
        prior_std: float = 0.5,
        values: Mapping[str, float] | None = None,
        only: Sequence[str] | None = None,
    ) -> tuple[YamlGroup, ...]:
        """The calibration groups of the declared parameters (``only`` these paths), with their sensitivities
        measured around ``values`` (nominal by default) on ``model`` (the model at ``values``, regenerated when not
        given); each group starts, and has its prior, at its value in ``values``."""
        values = dict(values or {})
        model = model or self.model(library, values)
        sensitivities = self.sensitivities(library, model, values)
        selected = [p for p in self.parameters if only is None or p.path in only]
        if only is not None and len(selected) != len(only):
            raise KeyError(f"Unknown calibration parameters: {sorted(set(only) - set(self.paths))}")
        return tuple(
            YamlGroup(
                name=parameter.path,
                value=values.get(parameter.path, parameter.value),
                lower=parameter.lower,
                upper=parameter.upper,
                exponents=sensitivities[parameter.path],
                prior_std=prior_std,
            )
            for parameter in selected
        )


__all__ = ["CalibrationSpec", "IncludeDefault", "YamlParameter"]
