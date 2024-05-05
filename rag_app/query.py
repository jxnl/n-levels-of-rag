import openai
from rag_app.models import TextChunk
from lancedb import connect
from typing import List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
import duckdb
import typer
from openai import AsyncOpenAI, OpenAI
import logfire
import instructor

app = typer.Typer()

openai_client = OpenAI()
logfire.instrument_openai(openai_client)

async_client = instructor.from_openai(openai_client)


@app.command(help="Query LanceDB for some results")
@logfire.instrument("Query", extract_args=True)
def db(
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
    query: str = typer.Option(help="Text to query against existing vector db chunks"),
    n: int = typer.Option(default=3, help="Maximum number of chunks to return"),
):
    logfire.configure()
    logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record="all"))
    with logfire.span("Create DB and Table"):
        if not Path(db_path).exists():
            raise ValueError(f"Database path {db_path} does not exist.")
        db = connect(db_path)
        db_table = db.open_table(table_name)

    with logfire.span("Embed using OpenAI text-embedding-3-large"):
        query_vector = (
            openai_client.embeddings.create(
                input=query, model="text-embedding-3-large", dimensions=256
            )
            .data[0]
            .embedding
        )
        logfire.info("Generated Query Embeddings", embeddings=query_vector)

    with logfire.span("Retrieving Results"):
        results: List[TextChunk] = (
            db_table.search(query_vector).limit(n).to_pydantic(TextChunk)
        )
        logfire.info("Generated Results", results=results)

    with logfire.span("Running Aggregation Query"):
        sql_table = db_table.to_lance()
        df = duckdb.query(
            "SELECT doc_id, count(chunk_id) as count FROM sql_table GROUP BY doc_id"
        ).to_df()

        doc_id_to_count = df.set_index("doc_id")["count"].to_dict()
        logfire.info("Generated counts", counts=doc_id_to_count)

    with logfire.span("Visualising Data"):
        table = Table(title="Results", box=box.HEAVY, padding=(1, 2), show_lines=True)
        table.add_column("Chunk Id", style="magenta")
        table.add_column("Content", style="magenta", max_width=120)
        table.add_column("Post Title", style="green", max_width=30)
        table.add_column("Chunk Number", style="yellow")
        table.add_column("Publish Date", style="blue")

        for result in results:
            chunk_number = f"{result.chunk_number}/{doc_id_to_count[result.doc_id]}"
            table.add_row(
                result.chunk_id,
                f"{result.post_title}({result.source})",
                result.text,
                chunk_number,
                result.publish_date.strftime("%Y-%m"),
            )
        Console().print(table)
