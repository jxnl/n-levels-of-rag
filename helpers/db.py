import lancedb
from helpers.models import TextChunk, Document
from typing import Iterable, Literal
from pathlib import Path


def create_db(path):
    db = lancedb.connect(path)
    return db


def create_table(
    db: lancedb.LanceDBConnection,
    table_name: str,
    mode: Literal["create", "overwrite"] = "overwrite",
) -> lancedb.db.LanceTable:
    return db.create_table(table_name, schema=TextChunk, mode=mode)


def read_files(path: Path, file_suffix: str) -> Iterable[Document]:
    for i, file in enumerate(path.iterdir()):
        if file.suffix != file_suffix:
            continue
        yield Document(id=i, text=file.read_text(), filename=file.name)


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


def batched(iterable, n=1):
    current_batch = []

    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch
