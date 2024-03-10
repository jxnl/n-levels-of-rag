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
    table.add_column("Post Title", style="green", max_width=30)
    table.add_column("Content", style="magenta", max_width=120)
    table.add_column("Chunk Number", style="yellow")
    table.add_column("Publish Date", style="blue")

    for result in results:
        chunk_number = f"{result.chunk_id}"
        table.add_row(
            f"{result.post_title}({result.source})",
            result.text,
            chunk_number,
            result.publish_date.strftime("%Y-%m"),
        )
    Console().print(table)
