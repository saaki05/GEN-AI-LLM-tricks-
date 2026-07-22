"""
Runs an experiment's program.py, capturing combined stdout/stderr to output.txt,
and rendering that captured text as a terminal-style screenshot.png in the same folder.

Usage: python scripts/run_and_capture.py <ExperimentFolder>
"""
import subprocess
import sys
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def render_terminal_screenshot(text, out_path, title="Terminal"):
    font = None
    for candidate in ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 16)
            break
    if font is None:
        font = ImageFont.load_default()

    wrapped_lines = []
    for line in text.splitlines():
        if line.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=110) or [""])

    max_lines = 60
    truncated = len(wrapped_lines) > max_lines
    if truncated:
        wrapped_lines = wrapped_lines[:max_lines] + ["... (truncated, see output.txt for full log) ..."]

    line_height = 20
    padding = 20
    header_height = 40
    width = 1000
    height = header_height + padding * 2 + line_height * len(wrapped_lines)

    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, header_height], fill=(45, 45, 45))
    draw.ellipse([15, 12, 31, 28], fill=(255, 95, 86))
    draw.ellipse([37, 12, 53, 28], fill=(255, 189, 46))
    draw.ellipse([59, 12, 75, 28], fill=(39, 201, 63))
    draw.text((90, 10), title, fill=(220, 220, 220), font=font)

    y = header_height + padding
    for line in wrapped_lines:
        draw.text((padding, y), line, fill=(200, 255, 200), font=font)
        y += line_height

    img.save(out_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_and_capture.py <ExperimentFolder>")
        sys.exit(1)

    exp_dir = ROOT / sys.argv[1]
    program = exp_dir / "program.py"
    if not program.exists():
        print(f"No program.py found in {exp_dir}")
        sys.exit(1)

    print(f"Running {program} ...")
    result = subprocess.run(
        [sys.executable, str(program)],
        cwd=str(exp_dir),
        capture_output=True,
        text=True,
    )

    combined = result.stdout
    if result.stderr:
        combined += "\n\n----- STDERR -----\n" + result.stderr

    output_txt = exp_dir / "output.txt"
    output_txt.write_text(combined, encoding="utf-8")

    render_terminal_screenshot(combined, exp_dir / "screenshot.png", title=exp_dir.name)

    print(f"Exit code: {result.returncode}")
    print(f"Saved: {output_txt}")
    print(f"Saved: {exp_dir / 'screenshot.png'}")
    if result.returncode != 0:
        print("!!! Script exited with an error. Check output.txt for details.")


if __name__ == "__main__":
    main()
