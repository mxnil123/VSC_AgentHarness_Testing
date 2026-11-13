import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_TESTS = ROOT / "tests" / "extension_host" / "index.js"


def _find_vscode() -> str:
    configured = os.environ.get("VSCODE_EXECUTABLE")
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            raise pytest.UsageError(
                f"VSCODE_EXECUTABLE does not name a file: {configured}"
            )
        if os.name == "nt" and candidate.name.lower() == "code.exe":
            cli = candidate.parent / "bin" / "code.cmd"
            if cli.is_file():
                return str(cli)
        return str(candidate)

    on_path = shutil.which("code")
    if on_path:
        return on_path

    if os.name == "nt":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft VS Code" / "bin" / "code.cmd",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    raise pytest.UsageError(
        "VS Code was not found. Install VS Code or set VSCODE_EXECUTABLE to its CLI."
    )


@pytest.mark.parametrize(
    "runtime_check",
    ["discovery", "token_count", "chat", "cancellation"],
    ids=["model-discovery", "token-counting", "chat-streaming", "cancellation"],
)
def test_custom_backend_through_real_vscode_extension_host(
    vscode_bridge, runtime_check
):
    executable = _find_vscode()
    with tempfile.TemporaryDirectory(
        prefix="vscode-llm-runtime-", ignore_cleanup_errors=True
    ) as temp_dir:
        temp = Path(temp_dir)
        (temp / "extensions").mkdir()
        results_path = temp / "results.json"
        command = [
            executable,
            "--no-sandbox",
            "--disable-gpu",
            "--skip-welcome",
            "--disable-workspace-trust",
            "--new-window",
            "--wait",
            f"--user-data-dir={temp / 'user-data'}",
            f"--extensions-dir={temp / 'extensions'}",
            f"--extensionDevelopmentPath={ROOT}",
            f"--extensionTestsPath={EXTENSION_TESTS}",
        ]
        environment = os.environ.copy()
        environment["LLM_HARNESS_BRIDGE_DIR"] = vscode_bridge["directory"]
        environment["LLM_HARNESS_TEST_RESULTS"] = str(results_path)
        environment["LLM_HARNESS_TEST_CASE"] = runtime_check
        timeout = int(environment.get("VSCODE_LLM_RUNTIME_TIMEOUT", "180"))
        initial_stats = vscode_bridge["bridge"].cancel_stats()

        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            pytest.fail(
                f"VS Code Extension Development Host timed out after {timeout}s.\n"
                f"stdout: {error.stdout}\nstderr: {error.stderr}"
            )

        runtime_logs = "\n".join(
            log.read_text(encoding="utf-8", errors="replace")
            for log in (temp / "user-data" / "logs").rglob("renderer.log")
        )
        output = f"{completed.stdout}\n{completed.stderr}\n{runtime_logs}"
        assert results_path.is_file(), (
            "The VS Code extension test runner did not write results. "
            f"Bridge stats: {vscode_bridge['bridge'].cancel_stats()}\n"
            f"Exit code: {completed.returncode}\n{output}"
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        test_results = results.get("tests", [])
        assert test_results, f"No VS Code runtime checks were reported.\n{output}"
        failures = [
            f"{item['name']}: {item['error']}"
            for item in test_results
            if not item["passed"]
        ]
        bridge_stats = vscode_bridge["bridge"].cancel_stats()
        models = vscode_bridge["bridge"].model_info()
        assert models, "VS Code discovery did not populate backend model metadata."
        model_ids = [model.id for model in models]
        assert len(model_ids) == len(set(model_ids)), (
            "The provider returned duplicate model IDs."
        )
        for model in models:
            assert model.id.strip() and model.name.strip()
            assert model.family.strip() and model.version.strip()
            assert model.max_input_tokens > 0 and model.max_output_tokens > 0
            assert isinstance(model.capabilities.image_input, bool)
            assert isinstance(model.capabilities.tool_calling, (bool, int))
            if isinstance(model.capabilities.tool_calling, int) and not isinstance(
                model.capabilities.tool_calling, bool
            ):
                assert model.capabilities.tool_calling >= 0
        assert completed.returncode == 0 and not failures, (
            "VS Code Extension Development Host checks failed.\n"
            + "\n".join(failures)
            + f"\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
        assert bridge_stats["model_discovery_requests"] > initial_stats[
            "model_discovery_requests"
        ], (
            "The VS Code host did not call the Python model discovery endpoint."
        )
        if runtime_check == "cancellation":
            assert bridge_stats["active_cancel_requests"] > initial_stats[
                "active_cancel_requests"
            ], (
                "The VS Code host did not cancel a request while the backend was active."
            )

        assert any(
            item["name"] == runtime_check and item["passed"]
            for item in test_results
        ), f"The requested runtime check {runtime_check!r} did not pass."
