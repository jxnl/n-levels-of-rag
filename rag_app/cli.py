import typer
import rag_app.query as QueryApp

app = typer.Typer(
    name="instructor-ft",
    help="A CLI for fine-tuning OpenAI's models",
)

app.add_typer(
    QueryApp.app, name="db", help="Commands to help interact with your local lancedb"
)
