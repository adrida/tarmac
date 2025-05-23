from typer.testing import CliRunner
from tarmaccore.cli import app


def test_version():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert "Tarmac v0.1-dev" in result.stdout
    assert result.exit_code == 0
