import tempfile
import threading
import time
import unittest
from pathlib import Path
from runtime_io import install


class WindowsPublicationTests(unittest.TestCase):
    def test_locked_reader_releases_and_atomic_write_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, source = root / 'worker_status.json', root / 'worker_status.json.tmp'
            target.write_text('old')
            source.write_text('new')
            reader = target.open('r')
            try:
                with self.assertRaises(PermissionError):
                    source.replace(target)
                original = install([root], timeout=2)
                release = threading.Thread(target=lambda: (time.sleep(0.15), reader.close()))
                release.start()
                try:
                    source.replace(target)
                    self.assertEqual(target.read_text(), 'new')
                    self.assertFalse(source.exists())
                finally:
                    Path.replace = original
                    release.join()
            finally:
                reader.close()

    def test_persistent_lock_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, source = root / 'status.json', root / 'status.tmp'
            target.write_text('old')
            source.write_text('new')
            with target.open('r'):
                original = install([root], timeout=0.1)
                try:
                    with self.assertRaises(PermissionError):
                        source.replace(target)
                    self.assertEqual(target.read_text(), 'old')
                    self.assertEqual(source.read_text(), 'new')
                finally:
                    Path.replace = original

    def test_other_io_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = install([root])
            try:
                with self.assertRaises(FileNotFoundError):
                    (root / 'missing.tmp').replace(root / 'status.json')
            finally:
                Path.replace = original


if __name__ == '__main__':
    unittest.main()
