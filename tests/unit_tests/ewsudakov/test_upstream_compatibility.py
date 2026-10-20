"""Retain upstream model metadata and process-specific f2py loading."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from madgraph import MG5DIR
from madgraph.iolibs.export_fks import ProcessExporterFortranFKS


class TestAlphaScheme(unittest.TestCase):
    def write_conversion(self, model, tagged=True):
        matrix_element = SimpleNamespace(
            born_me={
                "processes": [
                    {
                        "model": model,
                        "legs": [{"id": 22, "is_tagged": tagged}],
                    }
                ]
            }
        )
        output = io.StringIO()
        value = ProcessExporterFortranFKS.write_a0gmuconv_file(
            None, output, matrix_element
        )
        return value, output.getvalue()

    def test_alpha0_metadata_does_not_depend_on_model_directory(self):
        value, output = self.write_conversion(
            {"modelpath": "/models/renamed_model", "startfromalpha0": True}
        )
        self.assertTrue(value)
        self.assertIn("startfroma0=.true.", output)

    def test_directory_name_does_not_override_metadata(self):
        value, output = self.write_conversion(
            {"modelpath": "/models/loop_qcd_qed_sm_a0", "startfromalpha0": False}
        )
        self.assertFalse(value)
        self.assertIn("startfroma0=.false.", output)

    def test_untagged_photon_does_not_enable_conversion(self):
        value, _ = self.write_conversion({"startfromalpha0": True}, tagged=False)
        self.assertFalse(value)


class TestSudakovDispatcherImport(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("P0_uux_zz", "P0_ddx_zz"):
            (self.root / "SubProcesses" / name).mkdir(parents=True)
        template = (
            Path(MG5DIR) / "madgraph/iolibs/template_files/ewsudakov_pydispatcher.inc"
        ).read_text()
        self.namespace = {
            "__file__": str(self.root / "bin/internal/ewsud_pydispatcher.py")
        }
        old_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        exec(  # noqa: S102 - execute the generated dispatcher
            compile(
                template
                % {
                    "pdir_list": "",
                    "pdg2sud": "",
                    "pdgsorted": "",
                },
                self.namespace["__file__"],
                "exec",
            ),
            self.namespace,
        )

    def test_processes_load_distinct_native_modules(self):
        modules = {}

        def load(name):
            process = Path.cwd().name
            self.assertEqual(name, "ewsudpy_" + process)
            self.assertEqual(Path(sys.path[0]), Path.cwd())
            return modules.setdefault(name, object())

        before_cwd, before_path = os.getcwd(), sys.path[:]
        with mock.patch("importlib.import_module", side_effect=load):
            first = self.namespace["import_lib"]("P0_uux_zz")
            second = self.namespace["import_lib"]("P0_ddx_zz")
        self.assertIsNot(first, second)
        self.assertEqual(os.getcwd(), before_cwd)
        self.assertEqual(sys.path, before_path)

    def test_failed_import_restores_working_directory_and_search_path(self):
        before_cwd, before_path = os.getcwd(), sys.path[:]
        with (
            mock.patch("importlib.import_module", side_effect=ImportError),
            self.assertRaises(ImportError),
        ):
            self.namespace["import_lib"]("P0_uux_zz")
        self.assertEqual(os.getcwd(), before_cwd)
        self.assertEqual(sys.path, before_path)
