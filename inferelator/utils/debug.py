from __future__ import print_function, unicode_literals, division


class Debug:
    """
    This class is for printing status messages to stdout
    Just plain print doesn't work so well when there are multiple processes
    """
    verbose_level = 0
    default_level = 1

    silence_clients = True
    is_master = True

    levels = dict(silent=-1,
                  normal=0,
                  verbose=1, v=1,
                  very_verbose=2, vv=2,
                  max_output=3, vvv=3)

    @classmethod
    def set_verbose_level(cls, lvl):
        if isinstance(lvl, (int, float)):
            cls.verbose_level = lvl
        elif lvl in cls.levels.keys():
            cls.verbose_level = cls.levels[lvl]

    @classmethod
    def vprint(cls, *args, **kwargs):
        if cls.silence_clients and not cls.is_master:
            return
        cls.print_level(*args, **kwargs)

    @classmethod
    def allprint(cls, *args, **kwargs):
        cls.print_level(*args, **kwargs)

    @classmethod
    def print_level(cls, *args, **kwargs):
        try:
            level = kwargs.pop('level')
        except KeyError:
            level = cls.default_level
        if level <= cls.verbose_level:
            print((" " * level), *args, **kwargs)
        else:
            return
