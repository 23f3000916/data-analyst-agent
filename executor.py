def execute_code(code: str, globals_dict=None):
    """
    Execute Python code safely in a controlled environment.
    Ensures __name__ == '__main__' so main blocks are triggered.
    """
    if globals_dict is None:
        globals_dict = {"__name__": "__main__"}
    else:
        globals_dict.setdefault("__name__", "__main__")

    try:
        exec(code, globals_dict)
    except Exception as e:
        raise RuntimeError(f"Execution error: {e}")
    return globals_dict
