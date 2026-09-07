"""Small vendored runtime helper for YOLOP inference."""

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
        for item in self.generator:
            self.queue.put(item if self.preprocess_func is None else self.preprocess_func(item))
        self.queue.put(None)

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    next = __next__

    def __iter__(self):
        return self


def prefetch(max_prefetch=1):
    def decorating_function(function):
        def wrapper(*args, **kwargs):
            return BackgroundGenerator(function(*args, **kwargs), max_prefetch)

        return update_wrapper(wrapper, function)

    return decorating_function


background = prefetch
