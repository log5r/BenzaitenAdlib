"""Regression checks for imports, data paths, and relocated shell tools."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def run_python(self, code, cwd, **environment):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        env.pop('BENZAITEN_ROOT', None)
        env.update(environment)
        return subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env,
                              check=True, capture_output=True, text=True).stdout.strip()

    def test_paths_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python(
                'from benzaiten_adlib import paths; print(paths.ROOT)', directory)
            self.assertEqual(Path(result), ROOT)
            result = self.run_python(
                'from benzaiten_adlib import paths; paths.ensure_output_dirs(); '
                'print(paths.MODEL_DIR)', directory, BENZAITEN_ROOT=directory)
            self.assertEqual(Path(result), Path(directory).resolve() / 'models/current')
            for name in ('midi', 'solo', 'wav'):
                self.assertTrue((Path(directory) / 'output' / name).is_dir())

    def test_imports_do_not_train_generate_or_load_models(self):
        # Stub optional runtime libraries to isolate import behavior from ML setup.
        code = '''
import importlib
import sys
from unittest.mock import MagicMock
names = ['music21', 'music21.midi', 'numpy', 'matplotlib', 'matplotlib.pyplot',
         'mido', 'midi2audio', 'tensorflow', 'tensorflow_probability']
stubs = {name: MagicMock() for name in names}
class DistributionLambda:
    pass
stubs['tensorflow_probability'].layers.DistributionLambda = DistributionLambda
sys.modules.update(stubs)
for name in ['config', 'features', 'model_types', 'paths', 'submission',
             'music_utils', 'core', 'learn', 'generate']:
    importlib.import_module('benzaiten_adlib.' + name)
importlib.import_module('experiments.converter')
assert not any(stub.mock_calls for stub in stubs.values()), stubs
'''
        with tempfile.TemporaryDirectory() as directory:
            self.run_python(code, directory, BENZAITEN_ROOT=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(shutil.which('sh') and shutil.which('zip'), 'requires sh and zip')
    def test_shell_tools_and_source_archive(self):
        with tempfile.TemporaryDirectory(prefix='benzaiten test ') as directory:
            checkout = Path(directory) / 'checkout'
            checkout.mkdir()
            for name in ('benzaiten_adlib', 'scripts', 'experiments', 'tests', 'docs'):
                shutil.copytree(ROOT / name, checkout / name,
                                ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('LICENSE', 'README.md', 'README.ja.md', 'requirements.txt', 'pyproject.toml'):
                shutil.copy2(ROOT / name, checkout / name)
            def run_script(name):
                subprocess.run(['sh', str(checkout / 'scripts' / name)], cwd=directory,
                               check=True, capture_output=True, text=True)
            run_script('setup_required_folders.sh')
            keep = checkout / 'output/wav/keep.txt'
            keep.write_text('keep')
            for name, ext in [('midi', 'mid'), ('solo', 'mid'), ('wav', 'wav')]:
                (checkout / 'output' / name / f'test.{ext}').write_text('generated')
            run_script('remove_wav.sh')
            self.assertTrue(keep.exists())
            self.assertFalse(list((checkout / 'output').glob('*/*.mid')))
            self.assertFalse(list((checkout / 'output').glob('*/*.wav')))
            for _ in range(2):
                run_script('make_zip_of_code.sh')
            with zipfile.ZipFile(checkout / 'BenzaitenAdlibCode.zip') as archive:
                self.assertIsNone(archive.testzip())
                files = set(archive.namelist())
                for name in ('benzaiten_adlib/core.py', 'scripts/setup_required_folders.sh',
                             'experiments/converter.py', 'pyproject.toml',
                             'tests/test_project_layout.py'):
                    self.assertIn('BenzaitenAdlibCode/' + name, files)
                self.assertFalse(any('__pycache__' in name for name in files))
                archive.extractall(Path(directory) / 'unpacked')
            unpacked = Path(directory) / 'unpacked/BenzaitenAdlibCode'
            subprocess.run([sys.executable, '-c', 'import benzaiten_adlib.paths'],
                           cwd=unpacked, check=True)
            self.assertFalse((checkout / 'BenzaitenAdlibCode').exists())


if __name__ == '__main__':
    unittest.main()
