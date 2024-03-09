import typer
from lancedb import connect
from rag_app.models import TextChunk, Document
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
from rich import print

app = typer.Typer()


def read_files(path: Path, file_suffix: str) -> Iterable[Document]:
    for i, file in enumerate(path.iterdir()):
        if file.suffix != file_suffix:
            continue
        yield Document(id=i, text=file.read_text(), filename=file.name)


def batch_chunks(chunks, batch_size=20):
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


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


@app.command(help="Ingest data into a given lancedb")
def from_folder(
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
    folder_path: str = typer.Option(help="Folder to read data from"),
    file_suffix: str = typer.Option(default=".md", help="File suffix to filter by"),
):
    db = connect(db_path)

    if table_name not in db.table_names():
        db.create_table(table_name, schema=TextChunk, mode="overwrite")

    table = db.open_table(table_name)
    path = Path(folder_path)

    if not path.exists():
        raise ValueError(f"Ingestion folder of {folder_path} does not exist")

    files = read_files(path, file_suffix)
    chunks = chunk_text(files)
    batched_chunks = batch_chunks(chunks)

    ttl = 0
    for chunk_batch in tqdm(batched_chunks):
        table.add(chunk_batch)
        ttl += len(chunk_batch)

    print(f"Added {ttl} chunks to {table_name}")
