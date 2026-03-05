"""
overlay — quad-mapped content compositing system

Usage:

    from overlay import OverlayStack, BoxOverlay, Quad
    from overlay.effects import SolidContent

    stack = OverlayStack()

    content = SolidContent(color=(0, 200, 255))  # BGR yellow
    overlay = BoxOverlay(content, alpha=0.7)

    # Quad: (TL, TR, BR, BL) in normalized 0–1 coords
    quad: Quad = ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5))
    region_idx = overlay.add_region(quad)
    stack.add(overlay)

    # In your frame loop:
    #   stack.update(dt)
    #   frame = stack.render(frame)

    # Live-update a region's corners at any time:
    #   overlay.set_region(region_idx, new_quad)

Available content types:
    overlay.effects.SolidContent   — flat colour fill (implemented)
    overlay.video.VideoContent     — mp4 loop (stub)
    overlay.image.ImageContent     — static image (stub)
    overlay.filter.FilterContent   — wraps a source + transform fn (stub)
    overlay.effects.GlitchContent  — scan-line / channel glitch (stub)
"""

from .base import BoxContent, BoxOverlay, OverlayStack, Quad

__all__ = ["BoxContent", "BoxOverlay", "OverlayStack", "Quad"]
