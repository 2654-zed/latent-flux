import subprocess
import sys

monitor = subprocess.Popen(
    [sys.executable, "-m", "surveillance.deployment_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

routing = subprocess.Popen(
    [sys.executable, "-m", "surveillance.routing_monitor"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

# Keep alive — exit only if either process dies
monitor.wait()
routing.wait()
