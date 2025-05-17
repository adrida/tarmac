
import typer, rich
app = typer.Typer()
@app.command()
def version():
    rich.print("Tarmac v0.1-dev")
if __name__ == "__main__":
    app()
