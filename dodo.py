def task_fom():
    # instantiate example to define targets
    from src.getting_started.definitions import Example
    ex = Example(name="beam")
    return {
            "file_dep": ["./src/getting_started/fom.py"],
            "actions": ["python3 %(dependencies)s"],
            "targets": [ex.fine_grid],
            "clean": True,
            }
