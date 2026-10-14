"""Regression tests for the bundle packager's common-configuration roots."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py


SPEC = importlib.util.spec_from_file_location(
    "package_json_hdf5", Path(__file__).resolve().parents[1] / "tools/package_json_hdf5.py"
)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class CommonConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")

    def check_bundle(self, resource):
        mesh = self.root / resource
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"mesh at the runtime resource root")
        bundle = self.root / "input.h5"
        manifest = packager.package_scene(self.root / "input.json", bundle)
        self.assertIn("/" + resource, manifest)
        with h5py.File(bundle, "r") as hdf5:
            self.assertEqual(hdf5["/" + resource][()].tobytes(), mesh.read_bytes())
        return manifest

    def test_common_without_root_preserves_caller_root(self):
        self.write_json("input.json", {"common": "configs/common.json"})
        self.write_json("configs/common.json", {"geometry": {"mesh": "mesh.msh"}})
        manifest = self.check_bundle("mesh.msh")
        self.assertIn("/configs/common.json", manifest)
        self.assertNotIn("/configs/mesh.msh", manifest)

    def test_empty_common_root_preserves_caller_root(self):
        self.write_json("input.json", {"common": "configs/common.json"})
        self.write_json(
            "configs/common.json", {"root_path": "", "geometry": {"mesh": "mesh.msh"}}
        )
        self.check_bundle("mesh.msh")

    def test_explicit_common_root_overrides_caller_root(self):
        self.write_json("input.json", {"common": "configs/common.json"})
        self.write_json(
            "configs/common.json",
            {"root_path": "../assets", "geometry": {"mesh": "mesh.msh"}},
        )
        self.check_bundle("assets/mesh.msh")

    def test_common_preserves_explicit_input_root_and_patch(self):
        self.write_json(
            "input.json",
            {
                "root_path": "assets",
                "common": "configs/common.json",
                "patch": [{"op": "replace", "path": "/geometry/mesh", "value": "mesh.msh"}],
            },
        )
        self.write_json(
            "assets/configs/common.json", {"geometry": {"mesh": "original.msh"}}
        )
        self.check_bundle("assets/mesh.msh")

    def test_nested_common_is_located_relative_to_its_parent(self):
        self.write_json("input.json", {"common": "configs/common.json"})
        self.write_json("configs/common.json", {"common": "nested/base.json"})
        self.write_json(
            "configs/nested/base.json",
            {"root_path": "assets", "geometry": {"mesh": "mesh.msh"}},
        )
        manifest = self.check_bundle("mesh.msh")
        self.assertIn("/configs/nested/base.json", manifest)

    def test_explicit_roots_propagate_through_nested_common(self):
        self.write_json("input.json", {"common": "configs/common.json"})
        self.write_json(
            "configs/common.json", {"root_path": ".", "common": "nested/base.json"}
        )
        self.write_json(
            "configs/nested/base.json",
            {"root_path": "../assets", "geometry": {"mesh": "mesh.msh"}},
        )
        self.check_bundle("configs/assets/mesh.msh")


if __name__ == "__main__":
    unittest.main()
