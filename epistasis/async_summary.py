import json
import subprocess
import sys
import time

import psutil


class AsyncSummary:
    def __init__(self, root, output, script):
        self.root, self.output, self.script = root, output, script
        self.child = None
        matches = [p.info for p in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"], ad_value=None)
                   if p.info["cwd"] == str(root) and p.info["cmdline"] and script in p.info["cmdline"]]
        assert len(matches) <= 1, matches
        self.adopted = matches[0] if matches else None
        self.started = self.adopted["create_time"] if self.adopted else None
        self.published = json.loads((output / "manifest.json").read_text())["completed_evaluations"]

    def update(self, completed):
        if self.child is not None:
            code = self.child.poll()
            if code is None:
                return False
            assert code == 0, (self.script, code)
            self.child = None
        if self.adopted is not None:
            pid = self.adopted["pid"]
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if process.status() != psutil.STATUS_ZOMBIE:
                    assert process.create_time() == self.adopted["create_time"]
                    return False
            self.adopted = None
        manifest = self.output / "manifest.json"
        if self.started is not None:
            assert manifest.stat().st_mtime >= self.started, (self.script, "Summary exited before publishing its manifest")
            self.started = None
        self.published = json.loads(manifest.read_text())["completed_evaluations"]
        assert self.published <= completed
        if self.published == completed:
            return True
        log = self.output / "summary_completion.log"
        self.started = time.time()
        with log.open("a") as stream:
            self.child = subprocess.Popen([sys.executable, self.script], cwd=self.root,
                                          stdout=stream, stderr=subprocess.STDOUT)
        return False
