import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

LINE_PATTERN = re.compile(r"^\s*(\d+)\)\s*(\d+)\)\s*(dialog|text):\s*(.*)$")

def parse_line(line: str) -> Optional[Dict]:
    """
    Parse a raw line with format:
    0) 1) dialog: ...
    """
    line = line.strip()
    match = LINE_PATTERN.match(line)

    if not match:
        return None

    segment_id, scene_id, label, text = match.groups()

    return {
        "segment_id": int(segment_id),
        "scene_id": int(scene_id),
        "scene_key": f"{segment_id}_{scene_id}",
        "label": label,
        "text": text,
        "word_count": len(text.split()) if text else 0,
        "text_length": len(text),
    }

def extract_file_metadata(file_path: Path) -> Dict:
    """
    Extrae metadata desde la estructura real de rutas del dataset.
    """
    movie_id = file_path.parent.name
    character_name = file_path.stem

    if character_name.endswith("_text"):
        character_name = character_name[:-5]

    return {
        "movie_id": movie_id,
        "character_name": character_name,
        "source_file": str(file_path),
    }
def parse_character_file(file_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Parsea un archivo completo de personaje.

    Returns:
        records: líneas válidas parseadas
        errors: líneas inválidas con contexto
    """
    metadata = extract_file_metadata(file_path)
    records = []
    errors = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            parsed = parse_line(raw_line)

            if parsed is None:
                errors.append({
                    "source_file": str(file_path),
                    "line_number": line_number,
                    "raw_line": raw_line.rstrip("\n")
                })
                continue

            record = {
                **metadata,
                **parsed,
                "line_number": line_number,
            }
            records.append(record)

    return records, errors


def build_character_lines_dataset(file_paths: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida múltiples archivos en un único DataFrame.
    """
    all_records = []
    all_errors = []

    for file_path in file_paths:
        records, errors = parse_character_file(file_path)
        all_records.extend(records)
        all_errors.extend(errors)

    df = pd.DataFrame(all_records)
    errors_df = pd.DataFrame(all_errors)

    return df, errors_df