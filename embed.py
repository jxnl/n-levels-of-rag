from helpers.db import batched, chunk_text, create_db, create_table, read_files
import os
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path(os.getcwd(), "db")
DATA_PATH = Path(os.getcwd(), "data")


def main():
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
    db = create_db(DB_PATH)
    table = create_table(db, "paul_graham", "overwrite")
    documents = read_files(DATA_PATH, file_suffix=".md")
    chunks = chunk_text(documents)
    batched_chunks = batched(chunks, 20)

    for chunks in tqdm(batched_chunks):
        table.add(chunks)


if __name__ == "__main__":
    main()
