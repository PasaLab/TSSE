# -*- coding: utf-8 -*-
# @Time    : 05/12/2020 2:00 PM
# @File    : timetest.py


import sys
import inspect

PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2

if PY3:
    def _getargspec(func):
        return inspect.getfullargspec(func)


    def reraise(exc, tb=None):
        if exc.__traceback__ is not tb:
            raise exc.with_traceback(tb)
        raise exc

else:
    def _getargspec(func):
        return inspect.getargspec(func)

system_encoding = sys.getdefaultencoding()
if system_encoding == 'ascii':
    system_encoding = 'utf-8'


def exe_time(func):
    # return func
    import time

    def new_func(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back

    return new_func
