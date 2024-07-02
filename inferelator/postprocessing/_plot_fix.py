import matplotlib

# If matplotlib is being an idiot and trying to set a tkinter backend,
# switch to agg

# This is matplotlib 3.9+
try:
    if matplotlib.get_backend() in (
        i
        for i in matplotlib.backends.backend_registry.list_builtin(
            matplotlib.backends.BackendFilter.INTERACTIVE
        )
    ):
        matplotlib.use('agg')

# This is matplotlib <3.9
except AttributeError:

    if matplotlib.get_backend() in (
        i for i
        in matplotlib.rcsetup.interactive_bk
    ):
        matplotlib.use('agg')


import matplotlib.pyplot as plt
