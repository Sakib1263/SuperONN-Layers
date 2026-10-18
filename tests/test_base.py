"""Base import must remain framework-free, including in an installed wheel."""
import json
import subprocess
import sys
import unittest


class BaseTests(unittest.TestCase):
    def test_optional_imports(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys,json,superonn; from superonn.options import superonn_options; "
            "print(json.dumps([superonn.__version__,superonn_options(),"
            "'torch' in sys.modules,'tensorflow' in sys.modules]))"],
            text=True, capture_output=True, check=True, timeout=15)
        version, options, torch_loaded, tf_loaded = json.loads(result.stdout)
        self.assertEqual(options["q"], 3)
        self.assertEqual(options["output_chunk_size"], 4)
        self.assertTrue(options["recompute"])
        self.assertFalse(options["jit_compile"])
        self.assertFalse(torch_loaded)
        self.assertFalse(tf_loaded)

    def test_runner_timeout(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.run_checks",
                                 "base", "--test", "tests.test_base.BaseTests.test_optional_imports",
                                 "--seconds", ".001"],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["stopped"], "timeout")
