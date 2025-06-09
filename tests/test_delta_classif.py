from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import tempfile
import pathlib
from tarmac.cli import app
from typer.testing import CliRunner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings


def test_classif():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(max_iter=300).fit(X, y)
    rf = RandomForestClassifier().fit(X, y)
    p = pathlib.Path(tempfile.mkdtemp())
    joblib.dump(lr, p / "lr.pkl")
    joblib.dump(rf, p / "rf.pkl")
    res = CliRunner().invoke(
        app, ["diff", str(p / "lr.pkl"), str(p / "rf.pkl"), "--data", "iris"]
    )
    assert res.exit_code == 0
    assert "rules" in res.stdout


def test_classification_with_custom_data():

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=4, n_classes=3, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df_a = test_df.iloc[:100]
    test_df_b = test_df.iloc[100:200]

    p = pathlib.Path(tempfile.mkdtemp())

    test_df_a.drop("target", axis=1).to_csv(p / "Xa.csv", index=False)
    test_df_a[["target"]].to_csv(p / "ya.csv", index=False)
    test_df_b.drop("target", axis=1).to_csv(p / "Xb.csv", index=False)
    test_df_b[["target"]].to_csv(p / "yb.csv", index=False)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    model_a = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b = MLPClassifier(hidden_layer_sizes=(20, 10), random_state=42, max_iter=1000)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    joblib.dump(model_a, p / "model_a.pkl")
    joblib.dump(model_b, p / "model_b.pkl")

    runner = CliRunner()

    res_json = runner.invoke(
        app,
        [
            "diff",
            str(p / "model_a.pkl"),
            str(p / "model_b.pkl"),
            "--sampling",
            "union",
            "--Xa",
            str(p / "Xa.csv"),
            "--ya",
            str(p / "ya.csv"),
            "--Xb",
            str(p / "Xb.csv"),
            "--yb",
            str(p / "yb.csv"),
            "--task",
            "classification",
            "-o",
            str(p / "output.json"),
        ],
    )
    assert res_json.exit_code == 0
    assert "rules" in res_json.stdout
    assert len(res_json.stdout.split("feature")) > 2  # Ensure multiple rules

    res_txt = runner.invoke(
        app,
        [
            "diff",
            str(p / "model_a.pkl"),
            str(p / "model_b.pkl"),
            "--sampling",
            "union",
            "--Xa",
            str(p / "Xa.csv"),
            "--ya",
            str(p / "ya.csv"),
            "--Xb",
            str(p / "Xb.csv"),
            "--yb",
            str(p / "yb.csv"),
            "--task",
            "classification",
            "-o",
            str(p / "output.txt"),
        ],
    )
    assert res_txt.exit_code == 0
    assert "IF" in res_txt.stdout
    assert "THEN" in res_txt.stdout
    assert res_txt.stdout.count("IF") > 1  # Ensure multiple rules

    res_uf = runner.invoke(
        app,
        [
            "diff",
            str(p / "model_a.pkl"),
            str(p / "model_b.pkl"),
            "--sampling",
            "union",
            "--Xa",
            str(p / "Xa.csv"),
            "--ya",
            str(p / "ya.csv"),
            "--Xb",
            str(p / "Xb.csv"),
            "--yb",
            str(p / "yb.csv"),
            "--task",
            "classification",
            "--uf",
            "-o",
            str(p / "output_uf.txt"),
        ],
    )
    assert res_uf.exit_code == 0
    assert "Model Difference Analysis" in res_uf.stdout
    assert "explaining model differences:" in res_uf.stdout
    assert res_txt.stdout.count("IF") > 1  # Ensure multiple rules
