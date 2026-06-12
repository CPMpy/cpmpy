import logging
import sys

from ..runner.runner import Runner
from .base import Observer


class LoggerObserver(Observer):
    def __init__(self, **kwargs):
        # Use a unique logger name for this observer instance
        self.logger = logging.getLogger(f"{__name__}.LoggerObserver")
        # Set level to INFO to ensure messages are logged
        self.logger.setLevel(logging.INFO)
        # Disable propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        # Store reference to original stdout to always print there, even if redirected
        self.original_stdout = sys.__stdout__
        # Always add a new handler to ensure it writes to original stdout
        # Remove existing handlers first to avoid duplicates
        self.logger.handlers.clear()
        handler = logging.StreamHandler(self.original_stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Force the logger to be effective at INFO level
        self.logger.disabled = False

    def observe_init(self, runner: Runner):
        self.logger.info("Initializing runner")

    def observe_pre_transform(self, runner: Runner):
        self.logger.info("Pre-transforming")

    def observe_post_transform(self, runner: Runner):
        self.logger.info("Post-transforming")

    def observe_pre_solve(self, runner: Runner):
        self.logger.info("Pre-solving")

    def observe_post_solve(self, runner: Runner):
        self.logger.info("Post-solving")

    def print_comment(self, comment: str, runner: Runner):
        # Use info level to log comments
        self.logger.info(comment)
        # Also ensure it's flushed immediately
        for handler in self.logger.handlers:
            handler.flush()
