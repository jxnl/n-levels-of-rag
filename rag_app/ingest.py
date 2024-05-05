import typer
from lancedb import connect
from rag_app.models import TextChunk, Document
from pathlib import Path
from tqdm import tqdm
from rich import print
from rag_app.src.chunking import read_files, batch_items, chunk_text
import logfire

app = typer.Typer()

messages_sent = logfire.metric_counter("chunks_processed")


@app.command(help="Ingest data into a given lancedb")
@logfire.instrument("ingest_from_folder", extract_args=True)
def from_folder(
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
    folder_path: str = typer.Option(help="Folder to read data from"),
    file_suffix: str = typer.Option(default=".md", help="File suffix to filter by"),
):
    with logfire.span("Create DB and Table"):
        db = connect(db_path)

        if table_name not in db.table_names():
            db.create_table(table_name, schema=TextChunk, mode="overwrite")

        if "document" not in db.table_names():
            db.create_table("document", schema=Document, mode="overwrite")

    with logfire.span("Open Table and DB"):
        table = db.open_table(table_name)
        document_table = db.open_table("document")
        path = Path(folder_path)

        if not path.exists():
            raise ValueError(f"Ingestion folder of {folder_path} does not exist")

    with logfire.span("Load Documents into DB"):
        files = read_files(path, file_suffix)
        document_table.add(list(files))
        logfire.info("Extracted Metadata for Documents", documents=list(files))

    with logfire.span("Insert Chunks"):
        files = read_files(path, file_suffix=file_suffix)
        chunks = chunk_text(files)
        batched_chunks = batch_items(chunks)

        ttl = 0
        for idx, chunk_batch in tqdm(enumerate(batched_chunks)):
            with logfire.span(f"Processing batch {idx}"):
                table.add(chunk_batch)
                logfire.info(
                    "Inserted in batch of size {size}",
                    batch=chunk_batch,
                    size=len(chunk_batch),
                )
            ttl += len(chunk_batch)

    logfire.info("Added {ttl} chunks to {table_name}", ttl=ttl, table_name=table_name)
