"""Fix alpha modes in the converted Bistro glTFs (no Blender re-run needed).

Two misclassifications came out of convert_bistro_v2.py's alpha heuristic:

1. MASK materials whose base-color alpha is uniformly 0 (16x16 constant or
   zeroed channel): with alphaCutoff 0.5 every pixel discards and the whole
   mesh vanishes (table cloth, bottle labels, dirty-glass pane).  These carry
   no opacity information -> force OPAQUE.
2. Glass materials were left OPAQUE (their textures have alpha=1 everywhere),
   so window/bottle glass occluded everything behind it.  They become BLEND
   with a per-name constant alpha (frosted/frozen denser, painting glass
   thinner, bottles slightly tinted).
3. Blender's FBX importer marks most materials as blended (FBX transparency
   flags) and the exporter writes alphaMode=BLEND even when the effective
   alpha is 1.0 (tables, counters, walls...).  Those objects then render only
   in the transparency pass (no depth write, approximate sorting) and get
   eaten by geometry behind them.  Any BLEND material whose effective alpha
   is >= 0.95 is forced back to OPAQUE.

Usage: python scripts/fixup_bistro_alpha.py
Reads assets/bistro/*.gltf + the *_report.json alpha stats, edits in place.
Idempotent.
"""

import json
import os

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "bistro")
SCENES = ["BistroExterior", "BistroInterior"]

OPAQUE_ALPHA_THRESHOLD = 0.95


def glass_alpha(name):
    n = name.lower()
    if "frosted" in n or "frozen" in n:
        return 0.45
    if "paintings" in n:
        return 0.10
    if "liquorbottle" in n or "wine" in n:
        return 0.25
    return 0.18


def main():
    for scene in SCENES:
        gltf_path = os.path.join(ASSETS, scene + ".gltf")
        report_path = os.path.join(ASSETS, scene + "_report.json")
        report = json.load(open(report_path, encoding="utf-8"))
        zero_alpha = {m["material"] for m in report["materials"]
                      if m.get("alpha_lt_0.5_frac") == 1.0}

        j = json.load(open(gltf_path, encoding="utf-8"))
        n_opaque = n_blend = n_rescue = 0
        for m in j.get("materials", []):
            name = m.get("name", "")
            pbr = m.setdefault("pbrMetallicRoughness", {})
            factor = pbr.setdefault("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            if name in zero_alpha and m.get("alphaMode") == "MASK":
                # Uniform-zero alpha carries no cutout information.
                m.pop("alphaMode", None)
                m.pop("alphaCutoff", None)
                n_opaque += 1
                print("[%s] OPAQUE (was MASK, alpha==0): %s" % (scene, name))
            elif "glass" in name.lower():
                factor[3] = glass_alpha(name)
                m["alphaMode"] = "BLEND"
                m.pop("alphaCutoff", None)
                n_blend += 1
                print("[%s] BLEND a=%.2f: %s" % (scene, factor[3], name))
            elif m.get("alphaMode") == "BLEND" and factor[3] >= OPAQUE_ALPHA_THRESHOLD:
                # Blender/FBX export artifact: blended in name only.
                m.pop("alphaMode", None)
                n_rescue += 1
        print("[%s] %d BLEND(alpha>=%.2f) -> OPAQUE"
              % (scene, n_rescue, OPAQUE_ALPHA_THRESHOLD))

        with open(gltf_path, "w", encoding="utf-8") as f:
            json.dump(j, f, separators=(",", ":"))
        print("[%s] done: %d forced OPAQUE, %d glass -> BLEND"
              % (scene, n_opaque, n_blend))


if __name__ == "__main__":
    main()
