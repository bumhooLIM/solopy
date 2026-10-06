import logging
import tempfile
import unittest
from pathlib import Path

from solopy._logutil import get_logger


def _close_all(name):
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _count_in(path, text):
    return Path(path).read_text().count(text) if Path(path).exists() else 0


class TestGetLogger(unittest.TestCase):
    def setUp(self):
        self.name = f"solopy-test.{self.id()}"
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        for name in (self.name, "FitsLv2"):
            _close_all(name)
        self.tmp.cleanup()

    def test_repeated_calls_do_not_duplicate_handlers(self):
        log = self.dir / "a.log"
        for _ in range(3):
            logger = get_logger(self.name, log)
        self.assertEqual(sum(type(h) is logging.StreamHandler for h in logger.handlers), 1)
        self.assertEqual(sum(isinstance(h, logging.FileHandler) for h in logger.handlers), 1)
        self.assertFalse(logger.propagate)
        logger.info("hello")
        self.assertEqual(_count_in(log, "hello"), 1)

    def test_new_log_file_replaces_old_one(self):
        first, second = self.dir / "night1.log", self.dir / "night2.log"
        get_logger(self.name, first).info("one")
        logger = get_logger(self.name, second)
        logger.info("two")
        self.assertEqual(_count_in(first, "two"), 0)
        self.assertEqual(_count_in(second, "two"), 1)
        self.assertEqual(sum(isinstance(h, logging.FileHandler) for h in logger.handlers), 1)

    def test_none_keeps_existing_file_handler(self):
        log = self.dir / "keep.log"
        get_logger(self.name, log)
        get_logger(self.name, None).info("still here")
        self.assertEqual(_count_in(log, "still here"), 1)

    def test_two_fitslv2_instances_log_each_line_once(self):
        # Regression for primitive_repo.md §8 #5 (FitsLv3 creates a second FitsLv2).
        from solopy.fitslv2 import FitsLv2

        log = self.dir / "lv2.log"
        FitsLv2(log_file=log)
        lv2 = FitsLv2(log_file=log)
        lv2.logger.info("only once")
        self.assertEqual(_count_in(log, "only once"), 1)


if __name__ == "__main__":
    unittest.main()
