import sys
from contextlib import contextmanager

@contextmanager
def redirect_stdout(filepath):
    old_stdout = sys.stdout
    with open(filepath, 'w') as f:
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout
