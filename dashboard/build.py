"""Inject the run's real numbers into the console template."""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
data = pathlib.Path("pipeline_out/dashboard.json").read_text()
html = (ROOT / "template.html").read_text()
script = (ROOT / "console.js").read_text()
html = html.replace("__VMAX_DATA__", data).replace("__VMAX_SCRIPT__", script)
out = ROOT / "vmax_race_control.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1024:.0f} kB)")
