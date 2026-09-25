#!/usr/bin/env python3.11
"""Build the paper's system-contract figure as a dependency-free SVG."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs/current/figures/softwall_system_contract.svg"


def esc(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def text(x: int, y: int, value: str, size: int = 24, weight: int = 400,
         anchor: str = "middle", fill: str = "#202124") -> str:
    return (f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}">{esc(value)}</text>')


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = "#202124",
         radius: int = 12, dash: str = "") -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"{dashed}/>')


def line(x1: int, y1: int, x2: int, y2: int, color: str = "#37474f",
         width: float = 3, arrow: bool = True, dash: str = "") -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{marker}{dashed}/>')


parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="760" viewBox="0 0 1600 760">',
    '<defs>',
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#37474f"/></marker>',
    '<pattern id="recoveryHatch" width="12" height="12" patternUnits="userSpaceOnUse"><rect width="12" height="12" fill="#56b4a9"/><path d="M-2,2 l4,-4 M0,12 l12,-12 M10,14 l4,-4" stroke="#1b5e55" stroke-width="2"/></pattern>',
    '<pattern id="aiDots" width="12" height="12" patternUnits="userSpaceOnUse"><rect width="12" height="12" fill="#b39ddb"/><circle cx="3" cy="3" r="1.8" fill="#4a3475"/><circle cx="9" cy="9" r="1.8" fill="#4a3475"/></pattern>',
    '<style>.small{font:20px Arial,Helvetica,sans-serif;fill:#202124}.tiny{font:17px Arial,Helvetica,sans-serif;fill:#202124}.axis{stroke:#202124;stroke-width:2}.panel{fill:#ffffff;stroke:#9aa0a6;stroke-width:2}</style>',
    '</defs>',
    '<rect width="1600" height="760" fill="#ffffff"/>',
    rect(22, 34, 690, 690, "#ffffff", "#9aa0a6", 14),
    rect(736, 34, 842, 690, "#ffffff", "#9aa0a6", 14),
    text(54, 75, "(a) Certificate-preserving AI-RAN path", 27, 700, "start"),
    text(768, 75, "(b) One certificate covers all failure subsets", 27, 700, "start"),
]

# Panel A: system path.
parts += [
    rect(62, 125, 190, 92, "#d9ecff", "#2f5597"),
    text(157, 164, "RAN home 0", 24, 700),
    text(157, 193, "2 TB debts", 19),
    rect(62, 265, 190, 92, "#d9ecff", "#2f5597"),
    text(157, 304, "RAN home 1", 24, 700),
    text(157, 333, "2 TB debts", 19),
    rect(329, 112, 310, 118, "#ffe3b3", "#a65f00"),
    text(484, 150, "GPU 3: optional NeuralRx", 23, 700),
    text(484, 181, "actual TensorRT outcomes", 19),
    text(484, 207, "success deletes one debt", 19),
    rect(329, 270, 310, 138, "url(#recoveryHatch)", "#1b5e55"),
    text(484, 311, "GPU 2: shared cuPHY", 23, 700),
    text(484, 342, "conventional recovery lane", 19),
    text(484, 371, "CUDA IPC + NVLink P2P", 19),
    rect(329, 463, 310, 105, "url(#aiDots)", "#4a3475"),
    text(484, 505, "GPU 2: external AI", 23, 700),
    text(484, 537, "bounded Qwen work class", 19),
    rect(62, 455, 205, 145, "#eceff1", "#455a64"),
    text(165, 491, "SoftWall", 26, 700),
    text(165, 523, "global all-fail", 20),
    text(165, 550, "certificate", 20),
    text(165, 579, "+ independent verifier", 17),
    line(252, 166, 329, 166),
    line(252, 306, 329, 182),
    line(252, 196, 329, 317),
    line(252, 336, 329, 348),
    line(267, 505, 329, 358, "#455a64", 3),
    line(267, 527, 329, 510, "#455a64", 3),
    line(329, 548, 267, 560, "#455a64", 3),
    text(284, 437, "atomic calendar + AI lease", 17, 700),
    line(484, 230, 484, 270, "#a65f00", 3),
    text(505, 256, "outcome", 17, 400, "start"),
    rect(62, 640, 577, 52, "#f5f7f8", "#78909c", 8),
    text(350, 672, "Physical fence → credit retire → exactly one radio commit", 20, 700),
]

# Panel B: timeline helper.
x0, x1 = 795, 1518
scale = (x1 - x0) / 155.0

def tx(t: float) -> float:
    return x0 + t * scale


def block(t0: float, t1: float, y: int, h: int, fill: str, label: str,
          stroke: str = "#202124") -> None:
    parts.append(rect(int(tx(t0)), y, max(2, int((t1-t0)*scale)), h, fill, stroke, 4))
    parts.append(text(int((tx(t0)+tx(t1))/2), y+h//2+7, label, 17, 700))


parts += [
    text(780, 123, "Initial all-fail state: four unresolved debts", 22, 700, "start"),
    line(x0, 190, x1, 190, "#202124", 2, False),
]
for tick in (0, 45, 128, 155):
    parts.append(line(int(tx(tick)), 181, int(tx(tick)), 199, "#202124", 2, False))
    parts.append(text(int(tx(tick)), 221, str(tick), 16))
parts.append(text(x1+12, 196, "ms", 16, 400, "start"))
block(0, 45, 139, 38, "#ffe3b3", "NRx window", "#a65f00")
block(53, 78, 239, 45, "url(#recoveryHatch)", "R0", "#1b5e55")
block(78, 103, 239, 45, "url(#recoveryHatch)", "R1", "#1b5e55")
block(103, 128, 239, 45, "url(#recoveryHatch)", "R2", "#1b5e55")
block(128, 153, 239, 45, "url(#recoveryHatch)", "R3", "#1b5e55")
block(153, 155, 239, 45, "#d9dde0", "", "#616161")
parts += [
    text(780, 267, "all-fail executable certificate", 18, 700, "start"),
    text(780, 329, "After three observed successes: delete three credits atomically", 22, 700, "start"),
    line(x0, 397, x1, 397, "#202124", 2, False),
]
for tick in (0, 45, 88, 128, 155):
    parts.append(line(int(tx(tick)), 388, int(tx(tick)), 406, "#202124", 2, False))
    parts.append(text(int(tx(tick)), 428, str(tick), 16))
block(0, 45, 346, 38, "#ffe3b3", "NRx window", "#a65f00")
block(88, 128, 454, 48, "url(#aiDots)", "control 5 + Qwen 35", "#4a3475")
block(128, 153, 454, 48, "url(#recoveryHatch)", "R3", "#1b5e55")
block(153, 155, 454, 48, "#d9dde0", "", "#616161")
parts += [
    text(780, 483, "retimed certificate", 18, 700, "start"),
    rect(793, 558, 336, 102, "#e8f4ff", "#2f5597", 10),
    text(961, 596, "decision at 88 ms", 22, 700),
    text(961, 630, "40 + 25 + 2 = 67 ms → QSU", 19),
    rect(1170, 558, 346, 102, "#fff1e6", "#b64b00", 10),
    text(1343, 596, "decision at ≥89 ms", 22, 700),
    text(1343, 630, "40 + 25 + 2 > 66 ms → QSN", 19),
    text(1156, 700, "QSU/QSN boundary is predicted before physical execution", 22, 700),
    '</svg>',
]

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text("\n".join(parts) + "\n")
print(TARGET)
