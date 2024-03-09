import os
from pathlib import Path
from tqdm import tqdm
from lancedb import connect
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import Iterable


DB_PATH = Path(os.getcwd(), "db")
DATA_PATH = Path(os.getcwd(), "data")
DB_TABLE = "paul_graham"


class Document(BaseModel):
    id: int
    text: str
    filename: str


openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    id: int
    doc_id: int
    chunk_num: int
    start_pos: int
    end_pos: int
    text: str = openai.SourceField()
    # For some reason if we call openai.ndim(), it returns 1536 instead of 256 like we want
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)


def chunk_text(
    documents: Iterable[Document], window_size: int = 1024, overlap: int = 0
):
    id = 0
    for doc in documents:
        for chunk_num, start_pos in enumerate(
            range(0, len(doc.text), window_size - overlap)
        ):
            # TODO: Fix up this and use a Lance Model instead - have reached out to the team to ask for some help
            yield {
                "id": id,
                "doc_id": doc.id,
                "chunk_num": chunk_num,
                "start_pos": start_pos,
                "end_pos": start_pos + window_size,
                "text": doc.text[start_pos : start_pos + window_size],
            }
            id += 1


def read_file_content(path: Path, file_suffix: str) -> Iterable[Document]:
    for i, file in enumerate(path.iterdir()):
        if file.suffix != file_suffix:
            continue
        yield Document(id=i, text=file.read_text(), filename=file.name)


def batch_chunks(chunks, batch_size=10):
    batch = []
    for item in chunks:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def main():
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
    db = connect(DB_PATH)
    table = db.create_table(DB_TABLE, schema=TextChunk, mode="overwrite")

    documents = read_file_content(DATA_PATH, file_suffix=".md")
    chunks = chunk_text(documents)
    batched_chunks = batch_chunks(chunks, 20)

    for chunk_batch in tqdm(batched_chunks):
        table.add(chunk_batch)


if __name__ == "__main__":
    main()
