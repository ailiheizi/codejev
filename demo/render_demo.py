"""codejev principle demo — the full loop the author describes.

  1. the big model states the requirement
  2. it declares the target language and hands over that language's KEYWORD LIBRARY
     (the statement forms), plus the VARIABLES the requirement needs — the ones the
     keyword library does not contain
  3. host: candidate pool = language forms (parameterised by variables) + those variables
  4. the selector picks a form, then picks which variable/field fills each slot
  5. the host fills the slots and assembles the code
  6. the assembled result goes back to the big model to check

Note on honesty: this animation illustrates the design. The measured evidence covers the
candidate-page protocol (bench/jev_probe.py, bench/jev_batch_probe.py). Slot-level
composition of this exact shape is a design illustration, not a measured result.

All English. Pillow frames + ffmpeg. Output: demo/codejev-demo.mp4
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
FAINT = (52, 58, 66)
PANEL = (19, 24, 31)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
PURPLE = (188, 140, 255)
YELLOW = (210, 168, 32)
FONT = "/System/Library/Fonts/Menlo.ttc"

HERE = Path(__file__).resolve().parent
OUT = HERE / "codejev-demo.mp4"
FRAMES = HERE / "frames"

CHIP_H = 28
CHIP_GAP = 8
CAND_X, CAND_Y = 46, 196
OUT_X, OUT_Y = 690, 196


def f(size: int = 16) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


REQUIREMENT = "active_users(rows): keep active rows, return id and name"

LANGUAGE = "python"

# 大模型给出的语言关键词库：语句形式（带槽位）
KEYWORD_LIB = [
    ("def NAME(P):",                 "function header"),
    ("VAR = []",                     "initialise a list"),
    ("for VAR in VAR:",              "loop"),
    ("if VAR:",                      "filter"),
    ("VAR.append({...})",            "append a record"),
    ("return VAR",                   "return the list"),
]

# 大模型给出的变量（关键词库里没有的）
VARIABLES = ["rows", "active", "id", "name"]

# 选择器的答案（这里用真实跑出来的那种形状：一个 id 列表）
PICKS = [
    "def active_users(rows):",
    "result = []",
    "for row in rows:",
    'if row["active"]:',
    'result.append({"id": row["id"], "name": row["name"]})',
    "return result",
]


class Frame:
    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(self, x: float, y: float, s: str, color=FG, size: int = 16) -> None:
        self.d.text((int(x), int(y)), s, font=f(size), fill=color)

    def chip(self, x: float, y: float, s: str, color=FG, bg=(26, 32, 42),
             border=FAINT, bold: bool = False) -> None:
        xi, yi = int(x), int(y)
        w = len(s) * 9 + 22
        self.d.rounded_rectangle((xi, yi, xi + w, yi + CHIP_H), radius=5, fill=bg,
                                 outline=border, width=2 if bold else 1)
        self.text(xi + 11, yi + 6, s, color, 14)

    def save(self, i: int) -> None:
        self.img.save(FRAMES / f"{i:05d}.png")


def header(fr: Frame, step: str) -> None:
    fr.text(46, 32, "codejev", BLUE, 17)
    fr.text(124, 34, "the model only picks - the host assembles", DIM, 14)
    fr.text(W - 46 - len(step) * 8, 34, step, DIM, 14)
    fr.d.line([(46, 72), (W - 46, 72)], fill=FAINT)


def render() -> int:
    FRAMES.mkdir(exist_ok=True)
    index = 0

    def scene(n: int, fn) -> None:
        nonlocal index
        for k in range(n):
            fr = Frame()
            fn(fr, k / max(n - 1, 1))
            fr.save(index)
            index += 1

    # 1 — requirement
    def s1(fr: Frame, t: float) -> None:
        header(fr, "1 / 6")
        fr.text(46, 320, REQUIREMENT[: int(len(REQUIREMENT) * ease(t * 2))], FG, 21)

    scene(34, s1)

    # 2 — big model: language + keyword library + variables
    def s2(fr: Frame, t: float) -> None:
        header(fr, "2 / 6  big model: language + keyword library + variables")
        fr.text(46, 96, REQUIREMENT, DIM, 14)
        if ease(t * 3) > 0:
            fr.text(46, 132, f"target language: {LANGUAGE}", BLUE, 16)
        if ease(t * 3 - 0.4) > 0:
            fr.text(46, 168, "keyword library (statement forms of that language):", DIM, 14)
        for i, (form, note) in enumerate(KEYWORD_LIB):
            if ease(t * 3 - 0.7 - i * 0.14) <= 0:
                continue
            fr.chip(CAND_X + 16, 192 + i * (CHIP_H + CHIP_GAP), form, FG, (26, 32, 42))
            fr.text(CAND_X + 16 + len(form) * 9 + 34, 198 + i * (CHIP_H + CHIP_GAP), note, DIM, 13)
        if ease(t * 3 - 2.2) > 0:
            fr.text(46, 480, "variables the requirement needs", DIM, 14)
            fr.text(46, 502, "(not in the keyword library — the big model supplies them):",
                    DIM, 13)
        for j, var in enumerate(VARIABLES):
            if ease(t * 3 - 2.5 - j * 0.12) <= 0:
                continue
            fr.chip(46 + j * 110, 530, var, YELLOW, (40, 34, 22))

    scene(70, s2)

    # 3 — host builds the candidate pool from both
    def s3(fr: Frame, t: float) -> None:
        header(fr, "3 / 6  host: candidate pool = forms + variables")
        fr.text(46, 96, REQUIREMENT, DIM, 14)
        fr.text(CAND_X, CAND_Y - 30, "candidates", DIM, 14)
        for i, (form, _note) in enumerate(KEYWORD_LIB):
            if ease(t * 2.6 - i * 0.15) <= 0:
                continue
            fr.chip(CAND_X, CAND_Y + i * (CHIP_H + CHIP_GAP), form, FG)
        fr.d.line([(CAND_X, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 6),
                   (CAND_X + 330, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 6)], fill=FAINT)
        for j, var in enumerate(VARIABLES):
            if ease(t * 2.6 - 1.1 - j * 0.15) <= 0:
                continue
            fr.chip(CAND_X + j * 110, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 20, var, YELLOW,
                    (40, 34, 22))
        fr.text(OUT_X, OUT_Y - 30, "output", DIM, 14)
        fr.text(OUT_X, OUT_Y, "(empty)", (70, 76, 84), 14)

    scene(55, s3)

    # 4 — selector picks forms, slots get filled, code grows
    def s4(fr: Frame, t: float) -> None:
        header(fr, "4 / 6  selector picks a form; the host fills the slots")
        fr.text(46, 96, REQUIREMENT, DIM, 14)
        total = len(PICKS)
        pos = t * total
        done = int(pos)
        frac = pos - done

        fr.text(CAND_X, CAND_Y - 30, "candidates", DIM, 14)
        for i, (form, _note) in enumerate(KEYWORD_LIB):
            used = i < done
            fr.chip(CAND_X, CAND_Y + i * (CHIP_H + CHIP_GAP), form,
                    (90, 96, 104) if used else FG, PANEL if used else (26, 32, 42),
                    (40, 44, 52) if used else FAINT)
        fr.d.line([(CAND_X, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 6),
                   (CAND_X + 330, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 6)], fill=FAINT)
        for j, var in enumerate(VARIABLES):
            fr.chip(CAND_X + j * 110, CAND_Y + 6 * (CHIP_H + CHIP_GAP) + 20, var, YELLOW,
                    (40, 34, 22))

        fr.text(OUT_X, OUT_Y - 30, "assembled by the host", DIM, 14)
        for i in range(min(done, total)):
            fr.text(OUT_X, OUT_Y + i * 26, PICKS[i], GREEN, 14)
        if done < total:
            if frac < 0.5:
                fr.text(CAND_X, CAND_Y + 8 * (CHIP_H + CHIP_GAP) + 30,
                        f"selector -> form {done}", GREEN, 15)
            else:
                prog = ease((frac - 0.5) / 0.5)
                fr.chip(OUT_X, OUT_Y + done * 26 - (1 - prog) * 60, PICKS[done], GREEN,
                        (22, 44, 30), GREEN)

    scene(150, s4)

    # 5 — finished code
    def s5(fr: Frame, t: float) -> None:
        header(fr, "5 / 6  code assembled")
        fr.text(46, 96, REQUIREMENT, DIM, 14)
        for i, line in enumerate(PICKS):
            if ease(t * 3.2 - i * 0.14) <= 0:
                continue
            fr.text(OUT_X, OUT_Y + i * 26, line, GREEN, 14)
        if t > 0.5:
            fr.text(OUT_X, OUT_Y + 7 * 26, "every line came from a form the host enumerated", DIM, 14)
            fr.text(OUT_X, OUT_Y + 7 * 26 + 22, "and variables the big model supplied.", DIM, 14)

    scene(80, s5)

    # 6 — back to the big model
    def s6(fr: Frame, t: float) -> None:
        header(fr, "6 / 6  back to the big model")
        rows = [
            ("what goes back:", DIM, 15),
            ("  the requirement", FG, 15),
            ("  the chosen form ids and variable names", FG, 15),
            ("  the assembled code", FG, 15),
            ("  the diff against the file", FG, 15),
            ("", FG, 10),
            ("the big model decides: accept, re-instruct,", FG, 16),
            ("or supply a missing form/variable.", FG, 16),
        ]
        y = 180
        for i, (text, color, size) in enumerate(rows):
            if ease(t * 2.8 - i * 0.14) > 0:
                fr.text(70, y, text, color, size)
            y += 34 if text else 18
        if t > 0.75:
            fr.text(70, y + 10, "the selector never decides any of this.", DIM, 15)

    scene(85, s6)

    # 7 — principle + repo
    def s7(fr: Frame, t: float) -> None:
        header(fr, "")
        rows = [
            ("big model: requirement + language forms + variables", BLUE, 18),
            ("host: candidate pool  ->  selector: picks  ->  host: assembles", BLUE, 18),
            ("", FG, 12),
            ("the selector writes no code, chooses no path, approves nothing,", FG, 16),
            ("and never invents a name that is not in the pool.", FG, 16),
            ("", FG, 12),
            ("when a task is fixed enough, don't let the model", YELLOW, 17),
            ("write what it can choose.", YELLOW, 17),
            ("", FG, 12),
            ("github.com/ailiheizi/codejev", PURPLE, 16),
        ]
        y = 168
        for i, (text, color, size) in enumerate(rows):
            if ease(t * 3 - i * 0.12) > 0:
                fr.text(70, y, text, color, size)
            y += 34 if text else 18

    scene(85, s7)
    scene(55, lambda fr, _t: s7(fr, 1.0))
    return index


def main() -> int:
    print("rendering frames...")
    total = render()
    print(f"  {total} frames = {total / FPS:.1f}s")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", str(FRAMES / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-preset", "medium", str(OUT)],
        check=True,
    )
    print(f"  {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
