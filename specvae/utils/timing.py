#!/usr/bin/env python3

import time


def time_func(lg, arg1=None):
    """source: http://scottlobdell.me/2015/04/decorators-arguments-python/"""

    def real_decorator(function):

        def wrapper(*args, **kwargs):

            aa = arg1
            if aa is None:
                aa = function.__name__

            t1 = time.time()
            x = function(*args, **kwargs)
            t2 = time.time()
            lg.info("%s done %.02f s" % (aa, t2 - t1))
            return x

        return wrapper

    return real_decorator
