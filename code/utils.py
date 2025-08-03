import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Union, Optional

from paths import DATA_DIR, PUBLICATION_FPATH, ENV_FPATH, GUTENBERG_DATA_DIR


def load_gutenberg_books(book_id="pg3453"):
    """Loads the book text file.

    Returns:
        Content of the book as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    book_fpath = Path(os.path.join(GUTENBERG_DATA_DIR, f"{book_id}.txt"))

    # Check if file exists
    if not book_fpath.exists():
        raise FileNotFoundError(f"Book file not found: {book_fpath}")

    # Read and return the file content
    try:
        with open(book_fpath, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading book file: {e}") from e


def load_all_gutenberg_books(book_dir: str = GUTENBERG_DATA_DIR) -> list[str]:
    """Loads all the book text files in the given directory.

    Returns:
        List of books contents.
    """
    books = []
    for book_id in os.listdir(book_dir):
        if book_id.endswith(".txt"):
            books.append(load_gutenberg_books(book_id.replace(".txt", "")))
    return books


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env(api_key_type="GOOGLE_API_KEY") -> None:
    """Loads environment variables from a .env file and checks for required keys.

    Raises:
        AssertionError: If required keys are missing.
    """
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv(api_key_type)

    assert (
        api_key
    ), f"Environment variable '{api_key_type}' has not been loaded or is not set in the .env file."


def save_text_to_file(
    text: str, filepath: Union[str, Path], header: Optional[str] = None
) -> None:
    """Saves text content to a file, optionally with a header.

    Args:
        text: The content to write.
        filepath: Destination path for the file.
        header: Optional header text to include at the top.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)

    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e
