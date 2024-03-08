import typer
import openai
from rag_app.models import TextChunk
from lancedb import connect
import textwrap
from typing import List
from pathlib import Path

app = typer.Typer()


@app.command(help="Query LanceDB for some results")
def db(db_path: str, table_name: str, query: str, n: int = 3):
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

    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    table = Table(title="Results", box=box.HEAVY, padding=(1, 2), show_lines=True)
    table.add_column("Chunk ID", style="cyan", no_wrap=True)
    table.add_column("Result", style="magenta")

    for result in results:
        table.add_row(str(result.id), textwrap.fill(result.text, width=120))

    console.print(table)
