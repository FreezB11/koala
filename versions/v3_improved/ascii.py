"""
Prints the Koala wordmark with a depth-shaded orange gradient.

Uses true 24-bit ANSI color. Works in most modern terminals (iTerm2, Windows
Terminal, GNOME Terminal, kitty, alacritty, VS Code's integrated terminal...).
If your terminal doesn't support truecolor, see the fallback note at the bottom.
"""

import sys

# ---------- banner ----------
# This is a thin-line wordmark (built from _ | - characters, not block density
# glyphs), so depth here comes from heavy strokes (_) vs light strokes (|, -)
# rather than solid vs shaded blocks.

BANNER = r"""
 _____             _   ___ 
|  _  |___ ___ ___| |_|   |
|     | . | -_|   |  _| | |
|__|__|_  |___|_|_|_| |___|
      |___|                
"""

# ---------- color ramp ----------
# Heavy strokes (_) -> bright orange accent, foreground/closest.
# Light strokes (|, -) -> deep amber, background/receding.
BASE = "#5c2e0a"        # receding shade, deep amber/brown
BASE_DIM = "#8a4413"    # fallback for any stray glyph
ACCENT = "#ff8c1a"      # bright accent, foreground/closest
ACCENT_SOFT = "#e8761f"  # optional softer orange if you want less neon

RESET = "\033[0m"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fg(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"


def depth_color(ch: str) -> str:
    """Color every visible glyph the same bright orange (no shading)."""
    if ch.strip() == "":
        return None        # whitespace, no color needed
    return ACCENT


def render(banner: str, supports_color: bool) -> str:
    if not supports_color:
        return banner

    out_lines = []
    for line in banner.splitlines():
        rendered = []
        last_color = None
        for ch in line:
            color = depth_color(ch)
            if color is None:
                rendered.append(ch)
                last_color = None
                continue
            rgb = hex_to_rgb(color)
            code = fg(rgb)
            if code != last_color:
                rendered.append(code)
                last_color = code
            rendered.append(ch)
        rendered.append(RESET)
        out_lines.append("".join(rendered))
    return "\n".join(out_lines)


def terminal_supports_truecolor() -> bool:
    # Best-effort check; not bulletproof, but good enough for a banner.
    import os
    return sys.stdout.isatty() and os.environ.get("COLORTERM", "") in ("truecolor", "24bit")

def asciii():
    use_color = terminal_supports_truecolor() or "--force-color" in sys.argv
    print(render(BANNER, use_color))