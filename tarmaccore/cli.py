import typer
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
from pathlib import Path
from .adapters import get_adapter
from .delta.base import choose_builder
from .explainers.deltaxplainer import DeltaXplainer
from tarmaccore.data import load_table, union_datasets, load_builtin_dataset

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def callback(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit")
):
    if version:
        rprint("Tarmac v0.1-dev")
        raise typer.Exit()


@app.command()
def diff(
    model_a: Path,
    model_b: Path,
    sampling: str = typer.Option(
        "builtin",
        help="Sampling strategy: 'builtin' (iris/diabetes) or 'union'",
        show_default=True,
    ),
    data: str = typer.Option("iris", help="Which builtin dataset to use"),
    Xa: Path = typer.Option(
        None, "--Xa", help="Path to features for dataset A (CSV or NPY)"
    ),
    ya: Path = typer.Option(
        None, "--ya", help="Path to target for dataset A (CSV or NPY)"
    ),
    Xb: Path = typer.Option(
        None, "--Xb", help="Path to features for dataset B (CSV or NPY)"
    ),
    yb: Path = typer.Option(
        None, "--yb", help="Path to target for dataset B (CSV or NPY)"
    ),
    task: str = typer.Option("auto", help="classification|regression|auto"),
    epsilon: float = typer.Option(0.05, help="epsilon for regression Δ"),
    min_samples_leaf: float = typer.Option(
        0.01, help="Minimum samples per leaf (fraction of dataset)"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output file path (.json or .txt)"
    ),
    user_friendly: bool = typer.Option(
        False, "--uf", help="User friendly output format"
    ),
):
    """Compare two models and output rule explanations of their differences."""
    from sklearn import model_selection
    import json

    if sampling == "builtin":
        X, y = load_builtin_dataset(data)
    elif sampling == "union":
        if not (Xa and ya and Xb and yb):
            raise typer.BadParameter(
                "When using sampling='union', all of --Xa, --ya, --Xb, and --yb are required"
            )
        X_a = load_table(Xa)
        y_a = load_table(ya)
        X_b = load_table(Xb)
        y_b = load_table(yb)

        X, y = union_datasets(X_a, X_b, y_a, y_b)
    else:
        raise typer.BadParameter(f"Unknown sampling strategy: {sampling}")

    if y is not None:
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
            X, y, test_size=0.4, random_state=0
        )
    else:
        X_te = X

    load = get_adapter  # alias
    ma, mb = load(model_a), load(model_b)
    preds_a, preds_b = ma.predict(X_te), mb.predict(X_te)

    if task == "auto":
        task = "regression" if preds_a.dtype.kind in "f" else "classification"

    delta_labels = choose_builder(task).build(preds_a, preds_b, epsilon=epsilon)
    explainer = DeltaXplainer(min_leaf=min_samples_leaf).fit(X_te, delta_labels)
    rules = explainer.explain()

    console.print("\n[bold green]📊 Model Difference Analysis[/]")
    console.print(
        f"[bold blue]Generated {len(rules)} rules explaining model differences:[/]\n"
    )

    for i, rule in enumerate(rules[:10], 1):
        text = Text()
        text.append(f"Rule {i}: ", style="bold cyan")
        text.append(rule)
        console.print(Panel(text, expand=False))

    if output:
        if output.suffix == ".json":

            rules = explainer.explain(return_dict=True)

            output_dict = {
                "metadata": {
                    "total_rules": len(rules),
                    "task": task,
                    "epsilon": epsilon if task == "regression" else None,
                    "dataset_size": len(X_te),
                    "min_samples_leaf": min_samples_leaf,
                },
                "rules": rules,
            }
            with open(output, "w") as f:
                json.dump(output_dict, f, indent=2)
        elif output.suffix == ".txt":
            with open(output, "w") as f:
                if user_friendly:

                    f.write("📊 Analysis of Model Behavior Differences\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(
                        "This report identifies key patterns where the two models make different predictions.\n\n"
                    )
                    f.write(
                        f"We analyzed {len(X_te)} data samples and found {len(rules)} important patterns.\n"
                    )
                    f.write(
                        "Each pattern describes specific conditions where the models disagree.\n\n"
                    )
                    f.write("Key Findings:\n")
                    f.write("-" * 20 + "\n\n")
                    for i, rule in enumerate(rules, 1):
                        f.write(f"Pattern #{i}:\n")
                        f.write("What we found: When " + str(rule).lower() + "\n")
                        f.write(
                            "This means that under these specific conditions, the models produce notably different results.\n\n"
                        )
                    f.write(
                        "\nNote: Understanding these patterns can help identify where the models might need additional review or where their differences might impact business decisions.\n"
                    )
                else:

                    f.write(f"Dataset size: {len(X_te)} samples\n")
                    for i, rule in enumerate(rules, 1):
                        f.write(f"Rule {i}: {rule}\n")
        else:
            raise typer.BadParameter("Output file must have .json or .txt extension")


if __name__ == "__main__":
    app()
