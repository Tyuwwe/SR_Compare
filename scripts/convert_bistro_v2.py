"""Blender CLI script: convert Amazon Lumberyard Bistro FBX to glTF with
full PBR texture channels (baseColor+alpha, normal, ORM->MR+occlusion,
emissive, alpha MASK) and KHR_lights_punctual.

Usage (Git Bash):
  "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
      --background --factory-startup --python scripts/convert_bistro_v2.py -- \
      --input "D:/Code/Bistro_v5_2/BistroExterior.fbx" \
      --textures "D:/Code/Bistro_v5_2/Textures" \
      --output assets/bistro --name BistroExterior

Texture naming convention in the Bistro Textures dir:
  <base>_BaseColor.dds  RGB=base color, A=opacity (alpha test)
  <base>_Normal.dds     DirectX normal map (Y-), needs G flip for glTF
  <base>_Specular.dds   misnamed; actually ORM: R=AO, G=Roughness, B=Metalness
  <base>_Emissive.dds/.tga

Post-export (same Blender process):
  - flips the G channel of every exported *_Normal.png (DirectX -> glTF)
  - rewrites R=1 on exported ORM PNGs whose AO channel is empty (Bistro has
    no baked AO; R=0 would kill all ambient light once occlusion is injected)
  - injects occlusionTexture pointing at the metallicRoughness texture (ORM R=AO)
  - enforces alphaMode MASK + doubleSided for clip materials
Writes <output>/<name>_report.json with per-material channel coverage.
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np

# bistro_gltf_common.py lives next to this script. Blender --python does not
# always put the script directory on sys.path, so add it explicitly before
# importing the shared (non-bpy) helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bistro_gltf_common import (  # noqa: E402
    OPAQUE_ALPHA_THRESHOLD,
    ZERO_ALPHA_FRAC,
    glass_alpha,
)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--textures", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    return p.parse_args(argv)


def build_texture_index(tex_dir):
    """lowercased filename -> full path, for every file in the textures dir."""
    idx = {}
    for f in os.listdir(tex_dir):
        idx[f.lower()] = os.path.join(tex_dir, f)
    return idx


def find_basecolor_node(mat):
    """TexImage node feeding Principled 'Base Color'; fallback: any *_BaseColor image."""
    if not mat.use_nodes or not mat.node_tree:
        return None
    nt = mat.node_tree
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            inp = node.inputs.get("Base Color")
            if inp and inp.is_linked:
                src = inp.links[0].from_node
                if src.type == "TEX_IMAGE" and src.image:
                    return src
    for node in nt.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            if "_basecolor" in image_basename(node.image).lower():
                return node
    return None


def image_basename(img):
    """File basename without extension; falls back to the image datablock name."""
    p = bpy.path.abspath(img.filepath) if img.filepath else ""
    if p:
        return os.path.splitext(os.path.basename(p))[0]
    name = img.name
    for ext in (".dds", ".tga", ".png", ".jpg", ".tif"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    return name


def alpha_cutout_fraction(img):
    """Fraction of pixels with alpha < 0.5, or None when unavailable."""
    if img.size[0] == 0 or img.size[1] == 0:
        return None
    n = img.size[0] * img.size[1] * 4
    try:
        buf = np.empty(n, dtype=np.float32)
        img.pixels.foreach_get(buf)
    except Exception as e:
        print("[v2] WARNING: cannot read pixels of %s: %s" % (img.name, e))
        return None
    alpha = buf[3::4]
    return float(np.count_nonzero(alpha < 0.5)) / float(alpha.size)


def load_tex_image(nt, path, non_color, name):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.name = name
    node.label = name
    return node


def find_principled(mat):
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def set_input(node, name, value):
    inp = node.inputs.get(name)
    if inp is not None:
        inp.default_value = value
        return True
    return False


def process_material(mat, tex_idx, stats, clip_mats, blend_mats, normal_img_names,
                     orm_img_names):
    if not mat.use_nodes or not mat.node_tree:
        stats["no_nodes"] += 1
        return
    nt = mat.node_tree
    principled = find_principled(mat)
    if principled is None:
        stats["no_principled"] += 1
        return

    bc_node = find_basecolor_node(mat)
    entry = {"material": mat.name}
    if bc_node is None or bc_node.image is None:
        stats["no_basecolor_image"] += 1
        stats["materials"].append(entry)
        return

    bc_img = bc_node.image
    # Repair broken absolute paths coming from the FBX (original author paths).
    if bc_img.filepath:
        ap = bpy.path.abspath(bc_img.filepath)
        if not os.path.exists(ap):
            cand = tex_idx.get(os.path.basename(ap).lower())
            if cand:
                bc_img.filepath = cand
                try:
                    bc_img.reload()
                except Exception:
                    pass

    base = image_basename(bc_img)
    if base.lower().endswith("_basecolor"):
        base = base[: -len("_basecolor")]
    entry["base"] = base

    # ---- Normal map (DirectX Y-; flipped post-export) ---------------------
    npath = tex_idx.get((base + "_normal.dds").lower())
    if npath:
        node = load_tex_image(nt, npath, True, base + "_Normal")
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 1.0
        nt.links.new(node.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], principled.inputs["Normal"])
        entry["normal"] = True
        normal_img_names.add(os.path.splitext(os.path.basename(npath))[0])
        stats["normal"] += 1

    # ---- ORM ("_Specular.dds"): G=roughness, B=metalness ------------------
    spath = tex_idx.get((base + "_specular.dds").lower())
    if spath:
        node = load_tex_image(nt, spath, True, base + "_ORM")
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(node.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], principled.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], principled.inputs["Metallic"])
        entry["orm"] = True
        orm_img_names.add(os.path.splitext(os.path.basename(spath))[0])
        stats["orm"] += 1

    # ---- Emissive ---------------------------------------------------------
    epath = tex_idx.get((base + "_emissive.dds").lower()) or \
        tex_idx.get((base + "_emissive.tga").lower())
    if epath:
        node = load_tex_image(nt, epath, False, base + "_Emissive")
        em_in = principled.inputs.get("Emission Color") or \
            principled.inputs.get("Emission")
        if em_in is not None:
            nt.links.new(node.outputs["Color"], em_in)
            set_input(principled, "Emission Strength", 1.0)
            entry["emissive"] = True
            stats["emissive"] += 1

    # ---- Alpha cutout / glass ----------------------------------------------
    frac = alpha_cutout_fraction(bc_img)
    entry["alpha_lt_0.5_frac"] = frac
    if "glass" in mat.name.lower():
        # Transparent glass: BLEND with a per-name constant alpha (applied in
        # the JSON post-process); texture alpha carries no usable opacity.
        blend_mats[mat.name] = glass_alpha(mat.name)
        entry["alpha_mode"] = "BLEND"
        stats["blend"] += 1
    elif frac is not None and frac >= ZERO_ALPHA_FRAC:
        # Uniform-zero alpha carries no cutout information (16x16 constant or
        # zeroed channel); MASK would discard the entire mesh.
        entry["alpha_mode"] = "OPAQUE"
        mat.blend_method = "OPAQUE"
    elif frac is not None and frac > 0.01:
        try:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = 0.5
        except AttributeError:
            pass  # handled in JSON post-process regardless
        alpha_in = principled.inputs.get("Alpha")
        if alpha_in is not None and not alpha_in.is_linked:
            nt.links.new(bc_node.outputs["Alpha"], alpha_in)
        mat.use_backface_culling = False
        clip_mats.add(mat.name)
        entry["alpha_mode"] = "MASK"
        stats["mask"] += 1
    else:
        # FBX imports mark most materials as blended even when the effective
        # alpha is 1.0; force OPAQUE or they would render in the transparency
        # pass (no depth write) and lose the occlusion fight.
        entry["alpha_mode"] = "OPAQUE"
        mat.blend_method = "OPAQUE"

    stats["materials"].append(entry)


def flip_normal_pngs(out_dir, normal_img_names):
    """Invert the G channel of exported normal-map PNGs (DirectX -> glTF)."""
    flipped = 0
    for name in sorted(normal_img_names):
        path = os.path.join(out_dir, name + ".png")
        if not os.path.exists(path):
            print("[v2] WARNING: expected normal PNG missing: %s" % path)
            continue
        img = bpy.data.images.load(path, check_existing=False)
        try:
            n = img.size[0] * img.size[1] * 4
            buf = np.empty(n, dtype=np.float32)
            img.pixels.foreach_get(buf)
            buf[1::4] = 1.0 - buf[1::4]
            img.pixels.foreach_set(buf)
            img.save()
            flipped += 1
        except Exception as e:
            print("[v2] WARNING: normal flip failed for %s: %s" % (path, e))
        finally:
            bpy.data.images.remove(img)
    print("[v2] flipped G channel on %d normal PNGs" % flipped)
    return flipped


def fix_orm_ao_channel(out_dir, orm_img_names):
    """Neutralize the AO (R) channel of exported ORM PNGs when it holds no data.

    The Bistro '_Specular.dds' textures have R=0 everywhere (no AO baked --
    measured on the full set, 2026-08). glTF occlusion reads the R channel,
    so injecting occlusionTexture with R=0 would zero all ambient light.
    Where the R channel is empty it is rewritten to 1.0 (neutral AO); real
    AO data (if ever present) is kept. Returns (neutralized, kept)."""
    neutralized = 0
    kept = 0
    for name in sorted(orm_img_names):
        path = os.path.join(out_dir, name + ".png")
        if not os.path.exists(path):
            print("[v2] WARNING: expected ORM PNG missing: %s" % path)
            continue
        img = bpy.data.images.load(path, check_existing=False)
        try:
            n = img.size[0] * img.size[1] * 4
            buf = np.empty(n, dtype=np.float32)
            img.pixels.foreach_get(buf)
            r = buf[0::4]
            if float(r.mean()) < 0.01:
                buf[0::4] = 1.0
                img.pixels.foreach_set(buf)
                img.save()
                neutralized += 1
            else:
                kept += 1
        except Exception as e:
            print("[v2] WARNING: ORM AO fix failed for %s: %s" % (path, e))
        finally:
            bpy.data.images.remove(img)
    print("[v2] ORM AO channel: %d neutralized (R->1), %d with real AO kept"
          % (neutralized, kept))
    return neutralized, kept


def postprocess_gltf(gltf_path, clip_mats, blend_mats, zero_alpha_mats, out_dir):
    """Inject occlusionTexture (ORM R=AO), enforce MASK/doubleSided, apply
    BLEND glass, and rescue zero-alpha MASK materials to OPAQUE."""
    j = json.load(open(gltf_path, encoding="utf-8"))
    mats = j.get("materials", [])
    occ = 0
    mask = 0
    blend = 0
    missing_uri = []
    for m in mats:
        name = m.get("name")
        pbr = m.setdefault("pbrMetallicRoughness", {})
        mrt = pbr.get("metallicRoughnessTexture")
        if mrt is not None and "occlusionTexture" not in m:
            m["occlusionTexture"] = {"index": mrt["index"]}
            occ += 1
        if name in blend_mats:
            factor = pbr.setdefault("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            factor[3] = blend_mats[name]
            m["alphaMode"] = "BLEND"
            m.pop("alphaCutoff", None)
            blend += 1
        elif name in zero_alpha_mats:
            # Uniform-zero alpha: MASK would discard the whole mesh.
            m.pop("alphaMode", None)
            m.pop("alphaCutoff", None)
        elif name in clip_mats:
            m["alphaMode"] = "MASK"
            m["alphaCutoff"] = 0.5
            m["doubleSided"] = True
            mask += 1
        elif m.get("alphaMode") == "BLEND":
            # FBX/Blender export artifact (blended in name only, alpha == 1):
            # such meshes must stay in the opaque pass.
            pbr = m.get("pbrMetallicRoughness", {})
            if pbr.get("baseColorFactor", [1, 1, 1, 1])[3] >= OPAQUE_ALPHA_THRESHOLD:
                m.pop("alphaMode", None)
    em_boost = 0
    used = j.setdefault("extensionsUsed", [])
    for m in mats:
        name = (m.get("name") or "").lower()
        if "lantern" not in name and "streetlight" not in name and "emissive" not in name:
            continue
        m.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
            "emissiveStrength": 8.0
        }
        em_boost += 1
    if em_boost and "KHR_materials_emissive_strength" not in used:
        used.append("KHR_materials_emissive_strength")
    n_lights = 0
    ext = (j.get("extensions") or {}).get("KHR_lights_punctual") or {}
    n_lights = len(ext.get("lights") or [])
    # Verify every referenced image file exists.
    for img in j.get("images", []):
        uri = img.get("uri")
        if uri and not os.path.exists(os.path.join(out_dir, uri)):
            missing_uri.append(uri)
    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(j, f, separators=(",", ":"))
    print("[v2] post: occlusionTexture injected on %d, MASK on %d, BLEND on %d, "
          "emissive-boost %d, KHR_lights %d"
          % (occ, mask, blend, em_boost, n_lights))
    if missing_uri:
        print("[v2] WARNING: %d image URIs missing on disk: %s"
              % (len(missing_uri), missing_uri[:10]))
    return occ, mask, blend, missing_uri


def main():
    args = parse_args()
    # Blender resolves relative image paths against the .blend dir, not the
    # shell cwd; use an absolute output path for post-export file access.
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)
    tex_idx = build_texture_index(args.textures)
    print("[v2] texture index: %d files in %s" % (len(tex_idx), args.textures))

    bpy.ops.wm.read_factory_settings(use_empty=True)

    for fbx in args.input:
        print("[v2] importing %s" % fbx)
        # Native C++ importer: the legacy Python addon crashes on Blender 5.1
        # for FBX files containing lights.
        bpy.ops.wm.fbx_import(filepath=fbx)

    stats = {"normal": 0, "orm": 0, "emissive": 0, "mask": 0, "blend": 0,
             "no_nodes": 0, "no_principled": 0, "no_basecolor_image": 0,
             "materials": []}
    clip_mats = set()
    blend_mats = {}
    normal_img_names = set()
    orm_img_names = set()

    for mat in bpy.data.materials:
        process_material(mat, tex_idx, stats, clip_mats, blend_mats,
                         normal_img_names, orm_img_names)

    out_path = os.path.join(args.output, args.name + ".gltf")
    print("[v2] exporting %s" % out_path)
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLTF_SEPARATE",
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_yup=True,
        export_apply=False,
        export_animations=False,
        export_lights=True,  # KHR_lights_punctual (street lamps / strings)
    )

    flipped = flip_normal_pngs(args.output, normal_img_names)
    ao_neutral, ao_kept = fix_orm_ao_channel(args.output, orm_img_names)
    zero_alpha_mats = {e["material"] for e in stats["materials"]
                       if e.get("alpha_lt_0.5_frac") is not None
                       and e["alpha_lt_0.5_frac"] >= ZERO_ALPHA_FRAC}
    occ, mask, blend, missing_uri = postprocess_gltf(out_path, clip_mats, blend_mats,
                                                     zero_alpha_mats, args.output)

    j = json.load(open(out_path, encoding="utf-8"))
    report = {
        "scene": args.name,
        "materials_total": len(j.get("materials", [])),
        "images_total": len(j.get("images", [])),
        "textures_total": len(j.get("textures", [])),
        "channel_coverage": {
            "normal": stats["normal"],
            "orm_metallic_roughness": stats["orm"],
            "occlusion_injected": occ,
            "emissive": stats["emissive"],
            "alpha_mask": mask,
            "alpha_blend": blend,
        },
        "skipped": {
            "no_nodes": stats["no_nodes"],
            "no_principled": stats["no_principled"],
            "no_basecolor_image": stats["no_basecolor_image"],
        },
        "normal_pngs_flipped": flipped,
        "orm_ao_neutralized": ao_neutral,
        "orm_ao_kept": ao_kept,
        "lights_punctual": len(((j.get("extensions") or {}).get("KHR_lights_punctual") or {})
                               .get("lights") or []),
        "missing_image_uris": missing_uri,
        "materials": stats["materials"],
    }
    report_path = os.path.join(args.output, args.name + "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("[v2] report: %s" % report_path)
    print("[v2] SUMMARY %s: mats=%d images=%d normal=%d orm=%d occ=%d "
          "emissive=%d mask=%d"
          % (args.name, report["materials_total"], report["images_total"],
             stats["normal"], stats["orm"], occ, stats["emissive"], mask))
    print("[v2] done")


if __name__ == "__main__":
    main()
