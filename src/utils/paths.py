import pathlib

def get_project_root():
    """
    Returns the project root directory.
    Attempts to find the root by looking for the README.md file in parent directories.
    """
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "README.md").exists():
            return parent
    # Fallback to a fixed number of parents if README is not found
    return current.parents[1]

PROJECT_ROOT = get_project_root()
