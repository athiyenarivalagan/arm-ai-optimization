from pathlib import Path
import pandas as pd


# Dataset Path/s
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "ChnSentiCorp"
# DATASET_DIR = Path("../data/ChnSentiCorp")

def load_dataset(split: str) -> pd.DataFrame:
    # Dataset Loader
    valid_splits = {"train", "dev", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of {valid_splits}."
        )

    file_path = DATASET_DIR / f"{split}.tsv"

    df = pd.read_csv(
        file_path,
        sep="\t",
    )

    # validate_dataset(df)

    return df


# Dataset Validation
def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the ChnSentiCorp dataset format.
    """

    required_columns = [
        "label",
        "text_a",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )