"""The provider adapters, tested against a real HTTP server.

Mocking ``urllib`` would test that the adapters call the functions the test
expects them to call, which is not the question. The question is whether a
request goes out correctly shaped and a reply comes back correctly read, so
these tests run a local server that answers in each vendor's response format.
No network and no API key is involved.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mcdakit.ai_providers import (
    ProviderError,
    anthropic,
    google,
    load_env,
    ollama,
    openai,
    openai_compatible,
)


class _State:
    """What the fake server should do next, and what it saw."""

    def __init__(self):
        self.rate_limit_budget = 0
        self.calls = 0
        self.last_headers = {}
        self.last_body = {}


STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        """Silence the server's stderr logging."""

    def do_POST(self):
        STATE.calls += 1
        # HTTP header names are case-insensitive and urllib title-cases
        # them on the way out, so compare in lowercase.
        STATE.last_headers = {k.lower(): v for k, v in self.headers.items()}
        length = int(self.headers.get("Content-Length", 0))
        STATE.last_body = json.loads(self.rfile.read(length) or b"{}")

        if STATE.rate_limit_budget > 0:
            STATE.rate_limit_budget -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error": "rate limited"}')
            return

        if "/refuse" in self.path:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "malformed request"}')
            return

        if "/unexpected" in self.path:
            return self._json({"surprise": "not the shape you wanted"})

        if "chat/completions" in self.path:
            model = STATE.last_body.get("model", "?")
            return self._json({"choices": [{"message": {"content": model}}]})
        if "messages" in self.path:
            model = STATE.last_body.get("model", "?")
            return self._json({"content": [{"text": model}]})
        if "generateContent" in self.path:
            return self._json(
                {"candidates": [{"content": {"parts": [{"text": "gemini"}]}}]}
            )
        if "api/generate" in self.path:
            return self._json({"response": STATE.last_body.get("model", "?")})

        self.send_response(404)
        self.end_headers()
        return None

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def reset():
    STATE.rate_limit_budget = 0
    STATE.calls = 0


class TestEachProviderReadsItsOwnReplyShape:
    """The adapters exist because these four shapes are all different."""

    def test_openai(self, server):
        assert openai(api_key="k", base_url=f"{server}/v1")("hi") == "gpt-4o"

    def test_anthropic(self, server):
        ask = anthropic(api_key="k", base_url=f"{server}/v1")
        assert ask("hi") == "claude-sonnet-5"

    def test_google(self, server):
        assert google(api_key="k", base_url=f"{server}/v1")("hi") == "gemini"

    def test_ollama(self, server):
        assert ollama(base_url=server)("hi") == "llama3"

    def test_openai_compatible(self, server):
        ask = openai_compatible("mistral-7b", f"{server}/v1")
        assert ask("hi") == "mistral-7b"


class TestTheRequestIsShapedCorrectly:
    def test_anthropic_sends_its_required_headers(self, server):
        anthropic(api_key="secret", base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["x-api-key"] == "secret"
        assert STATE.last_headers["anthropic-version"] == "2023-06-01"

    def test_google_sends_its_key_header(self, server):
        google(api_key="secret", base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["x-goog-api-key"] == "secret"

    def test_openai_sends_a_bearer_token(self, server):
        openai(api_key="secret", base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["authorization"] == "Bearer secret"

    def test_a_local_server_needs_no_authorization(self, server):
        openai_compatible("m", f"{server}/v1")("hi")
        assert "authorization" not in STATE.last_headers

    def test_temperature_defaults_to_zero(self, server):
        """Sampling would make repeated asks differ for reasons unrelated to
        the model's confidence, which is what `samples` is measuring."""
        openai(api_key="k", base_url=f"{server}/v1")("hi")
        assert STATE.last_body["temperature"] == 0.0

    def test_the_prompt_is_sent_verbatim(self, server):
        openai(api_key="k", base_url=f"{server}/v1")("exact text")
        assert STATE.last_body["messages"][0]["content"] == "exact text"

    def test_ollama_does_not_stream(self, server):
        """A streamed reply arrives as many JSON objects, which json.loads
        cannot read."""
        ollama(base_url=server)("hi")
        assert STATE.last_body["stream"] is False


class TestRetry:
    def test_a_rate_limit_is_retried_and_survived(self, server):
        STATE.rate_limit_budget = 2
        assert openai(api_key="k", base_url=f"{server}/v1")("hi") == "gpt-4o"
        assert STATE.calls == 3

    def test_retries_are_bounded(self, server):
        STATE.rate_limit_budget = 99
        with pytest.raises(ProviderError, match="429"):
            openai(api_key="k", base_url=f"{server}/v1", retries=3)("hi")
        assert STATE.calls == 3

    def test_a_bad_request_is_not_retried(self, server):
        """A 400 will fail identically however often it is sent; retrying
        wastes the caller's time and the provider's quota."""
        with pytest.raises(ProviderError, match="400"):
            openai_compatible("m", f"{server}/refuse", retries=3)("hi")
        assert STATE.calls == 1

    def test_the_provider_message_survives(self, server):
        with pytest.raises(ProviderError, match="malformed request"):
            openai_compatible("m", f"{server}/refuse")("hi")


class TestFailuresAreLegible:
    def test_a_missing_key_says_which_variable_to_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
            anthropic()

    def test_a_missing_key_points_at_the_local_alternative(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ollama"):
            openai()

    def test_an_unreachable_server_is_named(self):
        with pytest.raises(ProviderError, match="Could not reach"):
            ollama(base_url="http://127.0.0.1:1", retries=1, timeout=2)("hi")

    def test_an_unexpected_shape_names_provider_and_path(self, server):
        with pytest.raises(ProviderError, match="Unexpected reply shape"):
            openai_compatible("m", f"{server}/unexpected")("hi")

    def test_an_unexpected_shape_shows_what_arrived(self, server):
        with pytest.raises(ProviderError, match="surprise"):
            openai_compatible("m", f"{server}/unexpected")("hi")


class TestKeysComeFromTheEnvironment:
    def test_openai_reads_its_variable(self, server, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        openai(base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["authorization"] == "Bearer from-env"

    def test_anthropic_reads_its_variable(self, server, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        anthropic(base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["x-api-key"] == "from-env"

    def test_google_accepts_either_variable(self, server, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
        google(base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["x-goog-api-key"] == "gemini-env"

    def test_an_explicit_key_wins(self, server, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        openai(api_key="explicit", base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["authorization"] == "Bearer explicit"


class TestTheyAreOrdinaryAskCallables:
    def test_a_provider_drives_the_ai_module(self, server):
        """The point of the adapters: they are accepted anywhere a
        hand-written `ask` is."""
        from mcdakit.ai import propose_criteria

        ask = openai_compatible("m", f"{server}/v1")
        # The fake server echoes the model name, so ask for a model whose
        # name is the JSON the parser expects.
        payload = '[{"name": "Price", "direction": "cost", "weight": 1.0}]'
        proposal = propose_criteria(
            "a problem", ask=openai_compatible(payload, f"{server}/v1")
        )
        assert [c.name for c in proposal.criteria] == ["Price"]
        assert callable(ask)

    def test_an_adapter_can_be_wrapped(self, server):
        """Nothing about them prevents logging, caching or rate limiting."""
        seen = []
        inner = openai_compatible("m", f"{server}/v1")

        def logged(prompt):
            seen.append(prompt)
            return inner(prompt)

        assert logged("hello") == "m"
        assert seen == ["hello"]


class TestNoSdkIsRequired:
    #: Everything the adapters are permitted to import. An allowlist rather
    #: than a check against `sys.stdlib_module_names`, which is Python 3.10+
    #: while this package supports 3.9 — and which would also silently admit
    #: any future stdlib addition. Adding a name here should be a decision.
    PERMITTED = frozenset(
        {"__future__", "collections", "json", "os", "pathlib", "time", "urllib"}
    )

    def test_the_module_imports_only_the_standard_library(self):
        """`pip install mcdakit` must be enough to reach every provider.

        A vendor SDK appearing here would forfeit the one-dependency claim,
        which the paper makes as a comparison against every rival library.
        """
        import ast
        import pathlib

        source = pathlib.Path(anthropic.__code__.co_filename).read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                imported.add(node.module.split(".")[0])

        unexpected = imported - self.PERMITTED
        assert not unexpected, (
            f"provider adapters import {unexpected}; every one must be in the "
            f"standard library, and adding to PERMITTED is a deliberate act"
        )

    def test_the_allowlist_is_really_the_standard_library(self):
        """Guards the guard: PERMITTED must not accumulate a third-party name.

        Skipped below 3.10, where the authoritative list does not exist.
        """
        import sys

        names = getattr(sys, "stdlib_module_names", None)
        if names is None:
            pytest.skip("sys.stdlib_module_names needs Python 3.10+")
        assert not (self.PERMITTED - set(names))


class TestBackoffAndTransportFailures:
    """Paths a happy-path test never reaches, each a real occurrence."""

    def test_a_connection_failure_is_retried_then_reported(self):
        """A local server starting up refuses connections for a moment."""
        import mcdakit.ai_providers as providers

        with pytest.raises(ProviderError, match="Could not reach"):
            providers._post(
                "http://127.0.0.1:1/nothing",
                {},
                {},
                timeout=1,
                retries=2,
                provider="Nowhere",
            )

    def test_retry_after_is_preferred_over_the_exponential_guess(self):
        """A provider knows when its own limit resets; we do not."""
        import urllib.error

        from mcdakit.ai_providers import _backoff

        error = urllib.error.HTTPError(
            "u", 429, "rate limited", {"Retry-After": "7"}, None
        )
        assert _backoff(error, attempt=0) == 7.0

    def test_an_absurd_retry_after_is_capped(self):
        """A day-long wait is never what a caller wants to sit through."""
        import urllib.error

        from mcdakit.ai_providers import _backoff

        error = urllib.error.HTTPError(
            "u", 429, "slow down", {"Retry-After": "86400"}, None
        )
        assert _backoff(error, attempt=0) == 60.0

    def test_an_unparsable_retry_after_falls_back(self):
        """The header may carry an HTTP date rather than seconds."""
        import urllib.error

        from mcdakit.ai_providers import _backoff

        error = urllib.error.HTTPError(
            "u", 429, "later", {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None
        )
        assert _backoff(error, attempt=2) == 4.0

    def test_no_retry_after_backs_off_exponentially(self):
        import urllib.error

        from mcdakit.ai_providers import _backoff

        error = urllib.error.HTTPError("u", 500, "boom", {}, None)
        assert _backoff(error, attempt=0) == 1.0
        assert _backoff(error, attempt=3) == 8.0

    def test_a_non_text_reply_is_refused(self):
        """Some providers return a content list where a string is expected;
        handing that to a JSON parser produces a baffling error later."""
        from mcdakit.ai_providers import _dig

        with pytest.raises(ProviderError, match=r"Expected text.*got list"):
            _dig(
                {"choices": [{"message": {"content": []}}]},
                ("choices", 0, "message", "content"),
                "TestCo",
            )

    def test_zero_retries_is_refused(self):
        """A request that is never attempted cannot succeed, and silently
        returning nothing would be worse than saying so."""
        import mcdakit.ai_providers as providers

        with pytest.raises(ProviderError, match="retries must be at least 1"):
            providers._post(
                "http://127.0.0.1:1/x", {}, {}, timeout=1, retries=0, provider="X"
            )


class TestLoadEnv:
    """Keys belong in a file that is never committed, not in source."""

    def write(self, tmp_path, text):
        path = tmp_path / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_it_sets_a_plain_assignment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        loaded = load_env(str(self.write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-x\n")))
        assert loaded == {"ANTHROPIC_API_KEY": "sk-ant-x"}
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-x"

    def test_a_provider_picks_the_key_up(self, tmp_path, monkeypatch, server):
        """The point of the whole exercise."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        load_env(str(self.write(tmp_path, "ANTHROPIC_API_KEY=from-file\n")))
        anthropic(base_url=f"{server}/v1")("hi")
        assert STATE.last_headers["x-api-key"] == "from-file"

    def test_it_understands_the_shell_forms(self, tmp_path, monkeypatch):
        """People copy these straight out of a terminal or a vendor's docs."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        loaded = load_env(
            str(
                self.write(
                    tmp_path,
                    "# a comment\n"
                    "\n"
                    "export OPENAI_API_KEY=sk-exported\n"
                    'GOOGLE_API_KEY="sk-double-quoted"\n'
                    "GEMINI_API_KEY='sk-single-quoted'\n"
                    "not a pair\n",
                )
            )
        )
        assert loaded["OPENAI_API_KEY"] == "sk-exported"
        assert loaded["GOOGLE_API_KEY"] == "sk-double-quoted"
        assert loaded["GEMINI_API_KEY"] == "sk-single-quoted"

    def test_the_real_environment_wins_by_default(self, tmp_path, monkeypatch):
        """A file in a checkout must not silently override what the deployment
        set, or running somewhere real becomes surprising."""
        monkeypatch.setenv("OPENAI_API_KEY", "from-deployment")
        loaded = load_env(str(self.write(tmp_path, "OPENAI_API_KEY=from-file\n")))
        assert os.environ["OPENAI_API_KEY"] == "from-deployment"
        assert loaded == {}

    def test_override_is_available_when_wanted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-deployment")
        load_env(str(self.write(tmp_path, "OPENAI_API_KEY=from-file\n")), override=True)
        assert os.environ["OPENAI_API_KEY"] == "from-file"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        """Production usually has no .env, and that is the normal case."""
        assert load_env(str(tmp_path / "nothing-here")) == {}

    def test_it_searches_upwards(self, tmp_path, monkeypatch):
        """So a script in a subdirectory finds the project's file."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=found-above\n")
        nested = tmp_path / "scripts" / "deep"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        assert load_env()["ANTHROPIC_API_KEY"] == "found-above"

    def test_a_value_may_contain_an_equals_sign(self, tmp_path, monkeypatch):
        """Base64-ish keys and URLs routinely do."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        loaded = load_env(str(self.write(tmp_path, "OPENAI_API_KEY=abc==def\n")))
        assert loaded["OPENAI_API_KEY"] == "abc==def"

    def test_a_blank_name_is_skipped(self, tmp_path):
        assert load_env(str(self.write(tmp_path, "=orphaned\n"))) == {}
