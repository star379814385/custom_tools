#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
from timeit import default_timer



__all__ = [
    "TimingError",
    "Timer",
]


class TimingError(RuntimeError):
    pass


class Timer(object):
    """A flexible Timer class

    Timing a function as decorator or a code block as context manager.

    Notes:
        Since we did not test it under multi-threading context, it's may
        thread-safe or not.

    Examples:

    >>> import time
    >>> from rrpyutils.timer import Timer
    >>> # use as decorator
    >>> @Timer()
    >>> def func():
    >>>     time.sleep(1)
    >>> func()
    func cost 1.0000 s
    >>> # or as context manager
    >>> with Timer('code block'):
    >>>     time.sleep(1)
    code block cost 1.0000 s
    >>> # disable timing directly without change your source code
    >>> Timer.disable()
    >>> func()
    >>> # also, enable
    >>> Timer.enable()
    >>> func()
    func cost 1.0000 s
    >>> timer1 = Timer()
    >>> timer1.start()
    >>> for i in range(3):
    >>>     time.sleep(i+1)
    >>>     timer1.trigger("try")
    >>>     # every interval_time between this time and last time
    >>>     print(timer1.interval_time)
    >>> time.sleep(0.1)
    >>>
    >>> print(timer1.trigger_time)
    >>> print(timer1.interval_time)
    >>> print(timer1.accumulate_interval_time)
    try cost 1.0008 s
    try cost 3.0036 s
    try cost 6.0062 s
    [21639.578154119, 21641.580766053, 21644.583136021]
    [1.0010871840022446, 2.002611934000015, 3.002369967998675]
    [1.0010871840022446, 3.0036991180022596, 6.0060690860009345]
    """

    _enable_timing = True

    @classmethod
    def enable(cls):
        cls._enable_timing = True

    @classmethod
    def disable(cls):
        cls._enable_timing = False

    @staticmethod
    def _empty_func(*_, **__):
        return "empty function return"

    def __init__(self, code_block_name=""):
        # attributes for trigger timer
        self._trigger_start = None
        self._trigger_time_list = []

        # attributes for context timer
        self._context_start = None
        self._context_block_name = code_block_name

    @staticmethod
    def _print_cost(msg, start, stop):
        elapsed_time = stop - start

        # logger.info(f"{msg} cost {elapsed_time:.4f} s")

    def _check_started(self):
        if self._trigger_start is None:
            raise TimingError("you should start timer first!")

    def start(self):
        self._trigger_start = default_timer()
        # restart
        self._trigger_time_list = []

    def trigger(self, msg=""):
        # record stop time first, since check may cost some time
        stop = default_timer()

        self._check_started()
        # record the system trigger time
        self._trigger_time_list.append(stop)

        self._print_cost(msg, self._trigger_start, stop)

    @property
    def trigger_time(self):
        self._check_started()

        return self._trigger_time_list

    @property
    def interval_time(self):
        interval_time_list = []

        self._check_started()
        for i in range(len(self._trigger_time_list)):
            if i == 0:
                interval_time_list.append(
                    self._trigger_time_list[i] - self._trigger_start
                )
            else:
                interval_time_list.append(
                    self._trigger_time_list[i] - self._trigger_time_list[i - 1]
                )

        return interval_time_list

    @property
    def accumulated_interval_time(self):
        accumulated_time_list = []

        self._check_started()
        for i in range(len(self._trigger_time_list)):
            accumulated_time_list.append(
                self._trigger_time_list[i] - self._trigger_start
            )

        return accumulated_time_list

    def __call__(self, func):
        msg = self._context_block_name or str(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not self._enable_timing:
                return func(*args, **kwargs)

            start = default_timer()

            ret = func(*args, **kwargs)

            stop = default_timer()

            self._print_cost(msg, start, stop)

            return ret

        return wrapped

    def __enter__(self):
        if self._enable_timing:
            self._context_start = default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enable_timing and exc_type is None:
            stop = default_timer()
            self._print_cost(self._context_block_name, self._context_start, stop)
