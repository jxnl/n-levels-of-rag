import typer
import openai
from rag_app.models import TextChunk
from lancedb import connect
import textwrap
from typing import List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer()


@app.command(help="Query LanceDB for some results")
def db(
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
    query: str = typer.Option(help="Text to query against existing vector db chunks"),
    n: int = typer.Option(default=3, help="Maximum number of chunks to return"),
):
    if not Path(db_path).exists():
        raise ValueError(f"Database path {db_path} does not exist.")
    db = connect(db_path)
    table = db.open_table(table_name)

    client = openai.OpenAI()
    query_vector = (
        client.embeddings.create(
            input=query, model="text-embedding-3-large", dimensions=256
        )
        .data[0]
        .embedding
    )

    results: List[TextChunk] = (
        table.search(query_vector).limit(n).to_pydantic(TextChunk)
    )

    table = Table(title="Results", box=box.HEAVY, padding=(1, 2), show_lines=True)
    table.add_column("Result", style="magenta")
    table.add_column("Post Title", style="green")
    table.add_column("Chunk Number", style="yellow")
    table.add_column("Publish Date", style="blue")

    for result in results:
        chunk_number = f"{result.chunk_id}/{result.post_chunk_count}"
        table.add_row(
            textwrap.fill(result.text, width=120),
            f"{result.post_title}\n({result.source})",
            chunk_number,
            result.publish_date.strftime("%Y-%m-%d"),
        )
    Console().print(table)
