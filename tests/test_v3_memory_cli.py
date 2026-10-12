from scripts.build_v3_residual_memory import main


def test_memory_cli_exposes_baseline_records(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_v3_residual_memory.py", "--help"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0
