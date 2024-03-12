import typer
from lancedb import connect
from rag_app.models import TextChunk, Document
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
from rich import print
import frontmatter
import hashlib
from datetime import datetime
from unstructured.partition.text import partition_text

app = typer.Typer()


def read_files(path: Path, file_suffix: str) -> Iterable[Document]:
    for i, file in enumerate(path.iterdir()):
        if file.suffix != file_suffix:
            continue
        post = frontmatter.load(file)
        yield Document(
            id=hashlib.md5(post.content.encode("utf-8")).hexdigest(),
            content=post.content,
            filename=file.name,
            metadata=post.metadata,
        )


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
    for doc in documents:
        for chunk_num, chunk in enumerate(partition_text(text=doc.content)):
            yield {
                "doc_id": doc.id,
                "chunk_id": chunk_num + 1,
                "text": chunk.text,
                "post_title": doc.metadata["title"],
                "publish_date": datetime.strptime(doc.metadata["date"], "%Y-%m"),
                "source": doc.metadata["url"],
            }


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
