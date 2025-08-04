def vprint(verbose, *args, **kwargs):
    """
    Print function that only prints if verbose is True.
    """
    import os
    if verbose:
        print(*args, **kwargs)


if __name__ == "__main__":
    vprint(True, "This is a verbose message.")
    vprint(False, "This message will not be printed.")
    vprint(False, "You can also control verbosity with keyword arguments.")
    vprint(True, "Verbose mode is enabled.")
    vprint(False, "This will always print because verbose is True by default.")