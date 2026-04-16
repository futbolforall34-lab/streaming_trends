from pathlib import Path
from typing import List
from src.utils.paths import PROJECT_ROOT

def get_character_texts_dir() -> Path:
    """
    Ruta esperada de los archivos de texto por personaje.
    """
    return PROJECT_ROOT / "data" / "raw" / "movie_characters" / "texts"

def list_character_files(extension: str = "*.txt") -> List[Path]:
    """
    Lista todos los archivos de personaje disponibles.
    """
    base_dir = get_character_texts_dir()
    return sorted(base_dir.rglob(extension))