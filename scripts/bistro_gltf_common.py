"""Shared Bistro glTF alpha-mode helpers.

Imported by both convert_bistro_v2.py (run inside Blender) and
fixup_bistro_alpha.py (run under plain Python), so this module MUST NOT import
bpy.
"""

# Uniform-zero alpha is detected with a small tolerance so that a 16x16
# constant/zeroed channel is treated as carrying no cutout information.
ZERO_ALPHA_FRAC = 0.999

# A BLEND material whose effective alpha is at least this is a Blender/FBX
# export artifact (blended in name only) and is forced back to OPAQUE.
OPAQUE_ALPHA_THRESHOLD = 0.95


def glass_alpha(name):
    """Constant alpha for BLEND glass (textures carry no usable opacity).

    Exterior shop windows are denser than interior / bottle glass so the dark
    unlit interior does not punch through as a black silhouette; SSR then
    reads as a mirror instead of a tinted hole.
    """
    n = name.lower()
    if "frosted" in n or "frozen" in n:
        return 0.45
    if "paintings" in n:
        return 0.10
    if "liquorbottle" in n or "wine" in n:
        return 0.25
    if "exterior" in n:
        return 0.55
    return 0.22
