import typer
import rag_app.query as QueryApp

app = typer.Typer(
    name="Rag-App",
    help="A CLI for querying a local RAG application backed by LanceDB",
)

app.add_typer(
    QueryApp.app,
    name="query",
    help="Commands to help query your local lancedb instance",
)
