#!/usr/bin/env python3
"""Package PolyFEM JSON inputs and their file dependencies into HDF5 bundles."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import posixpath
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def merge_patch(base: Any, patch: Any) -> Any:
    if not isinstance(patch, dict):
        return patch
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def pointer_tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer {pointer}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def pointer_parent(document: Any, pointer: str) -> tuple[Any, str]:
    tokens = pointer_tokens(pointer)
    if not tokens:
        raise ValueError("A root JSON pointer has no parent")
    parent = document
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    return parent, tokens[-1]


def pointer_get(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer_tokens(pointer):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def pointer_remove(document: Any, pointer: str) -> Any:
    parent, token = pointer_parent(document, pointer)
    return parent.pop(int(token)) if isinstance(parent, list) else parent.pop(token)


def pointer_add(document: Any, pointer: str, value: Any) -> None:
    parent, token = pointer_parent(document, pointer)
    if isinstance(parent, list):
        if token == "-":
            parent.append(value)
        else:
            parent.insert(int(token), value)
    else:
        parent[token] = value


def apply_json_patch(document: Any, operations: list[dict[str, Any]]) -> Any:
    result = copy.deepcopy(document)
    for operation in operations:
        kind = operation["op"]
        path = operation["path"]
        if kind == "remove":
            pointer_remove(result, path)
        elif kind == "add":
            pointer_add(result, path, copy.deepcopy(operation["value"]))
        elif kind == "replace":
            pointer_remove(result, path)
            pointer_add(result, path, copy.deepcopy(operation["value"]))
        elif kind == "move":
            pointer_add(result, path, pointer_remove(result, operation["from"]))
        elif kind == "copy":
            pointer_add(result, path, copy.deepcopy(pointer_get(result, operation["from"])))
        elif kind == "test":
            if pointer_get(result, path) != operation["value"]:
                raise RuntimeError(f"JSON patch test failed at {path}")
        else:
            raise ValueError(f"Unsupported JSON patch operation {kind}")
    return result


def logical_path(root: str, path: str) -> str:
    path = path.replace("\\", "/")
    joined = path if path.startswith("/") else posixpath.join(root, path)
    normalized = posixpath.normpath(joined)
    return normalized if normalized.startswith("/") else "/" + normalized


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf8") as stream:
        return json.load(stream)


class DependencyCollector:
    def __init__(self, input_path: Path):
        self.input_path = input_path.resolve()
        self.resources: dict[str, Path] = {}
        self.scanned_json: set[Path] = set()

    def add_file(self, logical: str, physical: Path) -> None:
        physical = physical.resolve()
        if not physical.is_file():
            raise FileNotFoundError(f"Resource {physical} does not exist")
        if logical in {"/config", "/json"} or logical.startswith("/_bundle/"):
            raise ValueError(f"Resource path {logical} is reserved by the HDF5 bundle format")
        previous = self.resources.get(logical)
        if previous is not None and previous != physical:
            raise RuntimeError(f"Logical resource {logical} maps to both {previous} and {physical}")
        self.resources[logical] = physical

    def resolve_common(
        self, config: dict[str, Any], physical_root: Path, logical_root: str
    ) -> tuple[dict[str, Any], Path, str]:
        if "common" not in config or not config["common"]:
            return config, physical_root, logical_root

        common_name = config["common"]
        if not isinstance(common_name, str):
            raise TypeError("The common field must be a string")
        common_file = (physical_root / common_name).resolve()
        common_logical = logical_path(logical_root, common_name)
        self.add_file(common_logical, common_file)
        common = read_json(common_file)
        if not isinstance(common, dict):
            raise TypeError(f"Common configuration {common_file} must contain an object")

        common_physical_root = common_file.parent
        common_logical_root = posixpath.dirname(common_logical) or "/"
        explicit_root = common.pop("root_path", "")
        if explicit_root:
            common_physical_root = (common_physical_root / explicit_root).resolve()
            common_logical_root = logical_path(common_logical_root, explicit_root)

        common, common_physical_root, common_logical_root = self.resolve_common(
            common, common_physical_root, common_logical_root
        )
        child = {key: value for key, value in config.items() if key not in {"common", "patch", "root_path"}}
        effective = merge_patch(common, child)
        if config.get("patch"):
            effective = apply_json_patch(effective, config["patch"])
        return effective, common_physical_root, common_logical_root

    def scan_json(self, value: Any, physical_root: Path, logical_root: str) -> None:
        if isinstance(value, dict):
            for child in value.values():
                self.scan_json(child, physical_root, logical_root)
            return
        if isinstance(value, list):
            for child in value:
                self.scan_json(child, physical_root, logical_root)
            return
        if not isinstance(value, str) or not value:
            return

        if any(character in value for character in "*?"):
            pattern = value if Path(value).is_absolute() else str(physical_root / value)
            for match in sorted(glob.glob(pattern, recursive=True)):
                physical = Path(match)
                if physical.is_file():
                    relative = physical.relative_to(physical_root).as_posix()
                    self.add_file(logical_path(logical_root, relative), physical)
            return

        physical = Path(value) if Path(value).is_absolute() else physical_root / value
        if physical.is_dir():
            directory_logical = logical_path(logical_root, value)
            for child in sorted(path for path in physical.rglob("*") if path.is_file()):
                self.add_file(
                    logical_path(directory_logical, child.relative_to(physical).as_posix()), child
                )
            return
        if not physical.is_file():
            return
        logical = logical_path(logical_root, value)
        self.add_file(logical, physical)

        resolved = physical.resolve()
        if physical.suffix.lower() == ".json" and resolved not in self.scanned_json:
            self.scanned_json.add(resolved)
            try:
                nested = read_json(resolved)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            self.scan_json(nested, resolved.parent, posixpath.dirname(logical) or "/")

    def collect(self) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
        config = read_json(self.input_path)
        if not isinstance(config, dict):
            raise TypeError(f"Input {self.input_path} must contain a JSON object")

        physical_root = self.input_path.parent
        logical_root = "/"
        explicit_root = config.get("root_path", "")
        if explicit_root:
            physical_root = (physical_root / explicit_root).resolve()
            logical_root = logical_path(logical_root, explicit_root)

        effective, physical_root, logical_root = self.resolve_common(
            dict(config), physical_root, logical_root
        )
        self.scan_json(effective, physical_root, logical_root)
        return config, sorted(self.resources.items())


def write_text_dataset(hdf5: h5py.File, path: str, contents: str) -> None:
    parent, _, name = path.rpartition("/")
    group = hdf5.require_group(parent or "/")
    if name in group:
        del group[name]
    group.create_dataset(name, data=contents, dtype=h5py.string_dtype(encoding="utf-8"))


def write_resource_dataset(hdf5: h5py.File, path: str, contents: bytes) -> None:
    parent, _, name = path.rpartition("/")
    group = hdf5.require_group(parent or "/")
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.frombuffer(contents, dtype=np.uint8))


def package_scene(input_path: Path, output_path: Path) -> list[str]:
    collector = DependencyCollector(input_path)
    config, resources = collector.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name("." + output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as hdf5:
            write_text_dataset(hdf5, "/config", json.dumps(config, separators=(",", ":")))
            manifest: list[str] = []
            for logical, physical in resources:
                write_resource_dataset(hdf5, logical, physical.read_bytes())
                manifest.append(logical)
            write_text_dataset(hdf5, "/_bundle/manifest", json.dumps(manifest))
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return [logical for logical, _ in resources]


def bundle_name(scene: str) -> str:
    return scene.replace("\\", "/").replace("/", "__").removesuffix(".json") + ".h5"


def package_manifest(manifest_path: Path, data_root: Path, output_dir: Path) -> None:
    manifest = read_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for entry in manifest["scenes"]:
        scene = entry["path"] if isinstance(entry, dict) else entry
        output = output_dir / bundle_name(scene)
        resources = package_scene(data_root / scene, output)
        generated.append({"path": scene, "bundle": output.name, "resources": resources})
        print(f"{scene}: {len(resources)} resources -> {output}")
    (output_dir / "manifest.json").write_text(json.dumps(generated, indent=2) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON input to package")
    source.add_argument("--manifest", type=Path, help="Manifest containing a scenes array")
    parser.add_argument("--output", type=Path, help="Output HDF5 path for --input")
    parser.add_argument("--data-root", type=Path, help="Scene root for --manifest")
    parser.add_argument("--output-dir", type=Path, help="Bundle directory for --manifest")
    args = parser.parse_args()

    if args.input:
        if args.output is None:
            parser.error("--input requires --output")
        package_scene(args.input, args.output)
    else:
        if args.data_root is None or args.output_dir is None:
            parser.error("--manifest requires --data-root and --output-dir")
        package_manifest(args.manifest, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
