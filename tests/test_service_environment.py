"""Exercise the real bot launcher against a tmux server with another bot's environment."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ServiceEnvironmentTest(unittest.TestCase):
    def test_bot_child_uses_resolved_credentials_and_preserves_caller_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executables = root / "bin"
            executables.mkdir()
            logs = root / "logs"
            logs.mkdir()
            report = root / "child.json"
            fake_bot = executables / "fake-bot"
            fake_bot.write_text("""#!/usr/bin/env python3
import json, os
from pathlib import Path
Path(os.environ['CHILD_REPORT']).write_text(json.dumps({
    'token': os.environ.get('TG_BOT_TOKEN'), 'env_file': os.environ.get('MINIMAX_ENV_FILE'),
    'allowlist': os.environ.get('TG_ALLOWLIST')}))
print('BOT_READY', flush=True)
""")
            fake_tmux = executables / "tmux"
            fake_tmux.write_text("""#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
if args[0] == 'has-session':
    sys.exit(1 if os.environ.get('NO_SESSION') == '1' else 0)
if args[0] == 'list-windows':
    sys.exit(0)
assert args[0] in ('new-window', 'new-session')
assert not any(token in ' '.join(args) for token in ('minimax-service-token', 'explicit-caller-token'))
from pathlib import Path
import stat
for index,value in enumerate(args[:-1]):
    if value == '-e' and args[index+1].startswith('MINIMAX_BOT_TOKEN_FILE='):
        assert stat.S_IMODE(Path(args[index+1].split('=',1)[1]).stat().st_mode) == 0o600
env = dict(os.environ, TG_BOT_TOKEN='other-service-token', TG_ALLOWLIST='wrong-service-user')
for index, value in enumerate(args[:-1]):
    if value == '-e':
        key, content = args[index + 1].split('=', 1)
        env[key] = content
sys.exit(subprocess.run(['bash', '-c', args[-1]], env=env).returncode)
""")
            fake_bot.chmod(0o700)
            fake_tmux.chmod(0o700)
            env_file = root / "service.env"
            env_file.write_text(f"TG_BOT_TOKEN=minimax-service-token\nTG_ALLOWLIST=42\n"
                                f"BOT_PYTHON='{fake_bot}'\nMINIMAX_LOG_DIR='{logs}'\n"
                                "MINIMAX_BOT_ENV_FILE=\nMINIMAX_ENABLE_TELEGRAM=1\n")
            env_file.chmod(0o600)
            script = Path(__file__).resolve().parents[1] / "bin/minimax-h3-service"
            environment = dict(os.environ, PATH=str(executables) + os.pathsep + os.environ["PATH"],
                               MINIMAX_ENV_FILE=str(env_file), CHILD_REPORT=str(report))
            for caller_token, expected, absent in ((None, "minimax-service-token", False),
                                           ("explicit-caller-token", "explicit-caller-token", False),
                                           (None, "minimax-service-token", True)):
                with self.subTest(explicit_override=caller_token is not None):
                    environment["NO_SESSION"] = "1" if absent else "0"
                    environment.pop("TG_BOT_TOKEN", None)
                    if caller_token is not None:
                        environment["TG_BOT_TOKEN"] = caller_token
                    result = subprocess.run(["bash", str(script), "bot-start"],
                                            env=environment, capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0)
                    child = json.loads(report.read_text())
                    self.assertEqual(child["token"], expected)
                    self.assertEqual(child["env_file"], str(env_file))
                    self.assertEqual(child["allowlist"], "42")
                    self.assertEqual(environment.get("TG_BOT_TOKEN"), caller_token)
                    self.assertNotIn(expected, result.stdout + result.stderr)
                    self.assertEqual(list((script.parents[1] / ".runtime").glob("bot-token.*")), [])


if __name__ == "__main__":
    unittest.main()
