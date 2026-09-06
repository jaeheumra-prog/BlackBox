"""Small vendored copy of the ``prefetch_generator`` runtime helper.

The official YOLOP utility imports :class:`BackgroundGenerator` at module
import time.  Keeping this tiny dependency beside the bundled YOLOP sources
ensures inference does not depend on an unlisted pip package in the runner.
"""

from functools import update_wrapper
from queue import Queue
from threading import Thread


class BackgroundGenerator(Thread):
    """Consume an iterator on a background thread with bounded prefetching."""

    def __init__(self, generator, max_prefetch=1, preprocess_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.preprocess_func = preprocess_func
        self.daemon = True
        self.start()

    def run(self):
        if self.preprocess_func is None:
            for item in self.generator:
                self.queue.put(item)
        else:
            for item in self.generator:
                self.queue.put(self.preprocess_func(item))
        self.queue.put(None)

    def next(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    __next__ = next

    def __iter__(self):
        return self


def prefetch(max_prefetch=1):
    """Decorator returning a :class:`BackgroundGenerator` for a generator."""

    def decorating_function(function):
        def wrapper(*args, **kwargs):
            return BackgroundGenerator(function(*args, **kwargs), max_prefetch)

        return update_wrapper(wrapper, function)

    return decorating_function


background = prefetch

