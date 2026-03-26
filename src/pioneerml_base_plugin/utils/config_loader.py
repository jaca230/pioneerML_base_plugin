from __future__ import annotations

from copy import deepcopy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_model_pipeline_config(model_name: str) -> dict[str, Any]:
    cfg_path = _package_root() / str(model_name) / "pipeline" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing plugin pipeline config: {cfg_path}")
    return dict(json.loads(cfg_path.read_text(encoding="utf-8")))


def load_full_chain_config() -> dict[str, Any]:
    cfg_path = _package_root() / "full_training_chain" / "pipeline" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing full-chain plugin pipeline config: {cfg_path}")
    return dict(json.loads(cfg_path.read_text(encoding="utf-8")))


def _set_loader_sources_in_place(
    cfg: dict[str, Any],
    *,
    main_sources: Sequence[str],
    optional_sources_by_name: Mapping[str, Sequence[str]] | None,
    source_type: str,
) -> None:
    loader_manager = cfg.get("loader_manager")
    if not isinstance(loader_manager, Mapping):
        return
    manager_cfg = loader_manager.get("config")
    if not isinstance(manager_cfg, Mapping):
        return

    spec = dict(manager_cfg.get("input_sources_spec") or {})
    spec["main_sources"] = [str(v) for v in list(main_sources)]
    spec["optional_sources_by_name"] = {
        str(name): [str(v) for v in list(values)]
        for name, values in dict(optional_sources_by_name or {}).items()
    }
    spec["source_type"] = str(source_type)
    manager_cfg = dict(manager_cfg)
    manager_cfg["input_sources_spec"] = spec

    loader_manager = dict(loader_manager)
    loader_manager["config"] = manager_cfg
    cfg["loader_manager"] = loader_manager


def _walk_and_patch_loader_sources(
    node: Any,
    *,
    main_sources: Sequence[str],
    optional_sources_by_name: Mapping[str, Sequence[str]] | None,
    source_type: str,
) -> Any:
    if isinstance(node, Mapping):
        out = dict(node)
        _set_loader_sources_in_place(
            out,
            main_sources=main_sources,
            optional_sources_by_name=optional_sources_by_name,
            source_type=source_type,
        )
        for key, value in list(out.items()):
            out[str(key)] = _walk_and_patch_loader_sources(
                value,
                main_sources=main_sources,
                optional_sources_by_name=optional_sources_by_name,
                source_type=source_type,
            )
        return out
    if isinstance(node, list):
        return [
            _walk_and_patch_loader_sources(
                item,
                main_sources=main_sources,
                optional_sources_by_name=optional_sources_by_name,
                source_type=source_type,
            )
            for item in node
        ]
    return node


def with_loader_sources(
    cfg: Mapping[str, Any],
    *,
    main_sources: Sequence[str],
    optional_sources_by_name: Mapping[str, Sequence[str]] | None = None,
    source_type: str = "file",
) -> dict[str, Any]:
    return dict(
        _walk_and_patch_loader_sources(
            deepcopy(dict(cfg)),
            main_sources=main_sources,
            optional_sources_by_name=optional_sources_by_name,
            source_type=source_type,
        )
    )


def with_model_handle_path(
    cfg: Mapping[str, Any],
    *,
    model_path: str | None,
) -> dict[str, Any]:
    out = deepcopy(dict(cfg))
    path_value = None if model_path in (None, "") else str(model_path)

    if isinstance(out.get("model_handle_builder"), Mapping):
        mh = dict(out["model_handle_builder"].get("model_handle") or {})
        mh_cfg = dict(mh.get("config") or {})
        mh_cfg["model_path"] = path_value
        out["model_handle_builder"] = {**dict(out["model_handle_builder"]), "model_handle": {**mh, "config": mh_cfg}}
    if isinstance(out.get("inference"), Mapping) and isinstance(out["inference"].get("model_handle_builder"), Mapping):
        inf = dict(out["inference"])
        mhb = dict(inf["model_handle_builder"])
        mh = dict(mhb.get("model_handle") or {})
        mh_cfg = dict(mh.get("config") or {})
        mh_cfg["model_path"] = path_value
        mhb["model_handle"] = {**mh, "config": mh_cfg}
        inf["model_handle_builder"] = mhb
        out["inference"] = inf
    return out


def with_writer_output(
    cfg: Mapping[str, Any],
    *,
    output_dir: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    out = deepcopy(dict(cfg))

    targets: list[dict[str, Any]] = []
    if isinstance(out.get("inference"), Mapping):
        inf = dict(out["inference"])
        if isinstance(inf.get("writer"), Mapping):
            targets.append(inf["writer"])
    if isinstance(out.get("writer"), Mapping):
        targets.append(out["writer"])

    for writer in targets:
        writer_cfg = dict(writer.get("config") or {})
        if output_dir is not None:
            writer_cfg["output_dir"] = str(output_dir)
        if output_path is not None:
            writer_cfg["output_path"] = str(output_path)
        writer["config"] = writer_cfg

    if isinstance(out.get("inference"), Mapping):
        inf = dict(out["inference"])
        if isinstance(inf.get("writer"), Mapping):
            inf["writer"] = targets[0] if targets else inf["writer"]
        out["inference"] = inf
    elif isinstance(out.get("writer"), Mapping) and targets:
        out["writer"] = targets[0]
    return out


def with_export_output(
    cfg: Mapping[str, Any],
    *,
    export_dir: str | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    out = deepcopy(dict(cfg))

    targets: list[dict[str, Any]] = []
    if isinstance(out.get("training"), Mapping):
        train = dict(out["training"])
        if isinstance(train.get("export"), Mapping):
            targets.append(train["export"])
    if isinstance(out.get("export"), Mapping):
        targets.append(out["export"])

    for export_stage in targets:
        exporter = export_stage.get("exporter")
        if not isinstance(exporter, Mapping):
            continue
        exporter_cfg = dict(exporter.get("config") or {})
        if export_dir is not None:
            exporter_cfg["export_dir"] = str(export_dir)
        if filename_prefix is not None:
            exporter_cfg["filename_prefix"] = str(filename_prefix)
        export_stage["exporter"] = {**dict(exporter), "config": exporter_cfg}

    if isinstance(out.get("training"), Mapping):
        train = dict(out["training"])
        if isinstance(train.get("export"), Mapping) and targets:
            train["export"] = targets[0]
        out["training"] = train
    elif isinstance(out.get("export"), Mapping) and targets:
        out["export"] = targets[0]

    return out
