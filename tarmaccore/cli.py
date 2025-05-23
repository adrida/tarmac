import typer
from rich import print as rprint

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def callback(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit")
):
    if version:
        rprint("Tarmac v0.1-dev")
        raise typer.Exit()


if __name__ == "__main__":
    app()
