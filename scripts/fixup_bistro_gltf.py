"""Post-process Bistro glTF exports for the sr_compare renderer limits.

The renderer (renderer/scene/GltfLoader.cpp + Renderer.cpp):
  - uploads EVERY glTF image into a texture array capped at 64 slots;
  - only reads baseColorTexture + baseColorFactor from materials.

So this script, for each input glTF:
  1. strips non-baseColor texture references (normal/specular/emissive/
     occlusion/metallicRoughness) -- the loader ignores them and they would
     waste texture slots;
  2. deduplicates baseColor images with identical PNG content;
  3. if more than --max-textures unique baseColor images remain, keeps the
     --max-textures most-used ones (by triangle count of referencing
     primitives) and bakes the average color of each dropped texture into the
     material's baseColorFactor instead;
  4. rewrites images/textures arrays;
  5. after ALL inputs are processed, deletes PNGs not referenced by any of
     them (safe when several glTFs share one texture directory).

Usage: python scripts/fixup_bistro_gltf.py assets/bistro/BistroExterior.gltf \
          assets/bistro/BistroInterior.gltf
"""

import argparse
import hashlib
import json
import os
import struct
import zlib


def png_average_rgb(path):
    """Average RGB of a PNG (handles 8-bit RGB/RGBA, non-interlaced)."""
    if not os.path.exists(path):
        return None
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    width = height = bitdepth = colortype = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, bitdepth, colortype = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if bitdepth != 8 or colortype not in (2, 6):
        return None
    raw = zlib.decompress(bytes(idat))
    bpp = 3 if colortype == 2 else 4
    stride = width * bpp
    prev = bytearray(stride)
    rs = gs = bs = count = 0
    off = 0
    for _ in range(height):
        f = raw[off]
        line = bytearray(raw[off + 1:off + 1 + stride])
        off += 1 + stride
        if f == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        for i in range(0, stride, bpp):
            rs += line[i]; gs += line[i + 1]; bs += line[i + 2]
        count += width
        prev = line
    if count == 0:
        return None
    return (rs / count / 255.0, gs / count / 255.0, bs / count / 255.0)


def process(gltf_path, max_textures):
    """Rewrite one glTF; return the set of texture URIs it still references."""
    gdir = os.path.dirname(gltf_path)
    j = json.load(open(gltf_path, encoding="utf-8"))
    mats = j.get("materials", [])
    images = j.get("images", [])
    textures = j.get("textures", [])
    acc = j.get("accessors", [])

    # 1. Strip everything except baseColorTexture from materials.
    for m in mats:
        pbr = m.get("pbrMetallicRoughness", {})
        pbr.pop("metallicRoughnessTexture", None)
        m.pop("normalTexture", None)
        m.pop("occlusionTexture", None)
        m.pop("emissiveTexture", None)
        for ext in (m.get("extensions") or {}).values():
            if isinstance(ext, dict):
                for k in list(ext.keys()):
                    if k.endswith("Texture"):
                        ext.pop(k, None)

    # baseColor texture index per material.
    mat_tex = []
    for m in mats:
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        mat_tex.append(bct["index"] if bct else None)

    # 2. Usage (triangle count) per texture index.
    tri_usage = {}
    for mesh in j.get("meshes", []):
        for prim in mesh.get("primitives", []):
            mi = prim.get("material")
            if mi is None or mi >= len(mat_tex):
                continue
            ti = mat_tex[mi]
            if ti is None:
                continue
            tris = (acc[prim["indices"]]["count"] // 3) if "indices" in prim \
                else acc[prim["attributes"]["POSITION"]]["count"] // 3
            tri_usage[ti] = tri_usage.get(ti, 0) + tris

    used_tex = sorted(tri_usage.keys())

    # 3. Dedupe by content.
    canon = {}   # used-tex index -> canonical used-tex index
    by_hash = {}
    for t in used_tex:
        uri = images[textures[t]["source"]].get("uri")
        h = None
        if uri:
            p = os.path.join(gdir, uri)
            if os.path.exists(p):
                h = hashlib.md5(open(p, "rb").read()).hexdigest()
        key = h if h else ("__unique__", t)
        if key in by_hash:
            canon[t] = by_hash[key]
            tri_usage[by_hash[key]] = tri_usage.get(by_hash[key], 0) + tri_usage[t]
        else:
            by_hash[key] = t
            canon[t] = t

    unique_tex = sorted(set(canon.values()), key=lambda t: -tri_usage.get(t, 0))
    print("[fixup] %s: %d used baseColor textures, %d unique contents"
          % (gltf_path, len(used_tex), len(unique_tex)))

    # 4. Keep top max_textures; bake average color for the rest.
    keep = set(unique_tex[:max_textures])
    for t in unique_tex:
        if t in keep:
            continue
        uri = images[textures[t]["source"]].get("uri")
        avg = png_average_rgb(os.path.join(gdir, uri)) if uri else None
        for mi, ti in enumerate(mat_tex):
            if ti is not None and canon.get(ti) == t:
                m = mats[mi]
                pbr = m.setdefault("pbrMetallicRoughness", {})
                pbr.pop("baseColorTexture", None)
                if avg:
                    f = pbr.get("baseColorFactor", [1, 1, 1, 1])
                    pbr["baseColorFactor"] = [f[0] * avg[0], f[1] * avg[1],
                                              f[2] * avg[2], f[3]]
                mat_tex[mi] = None
    print("[fixup] kept %d textures, baked average color for %d"
          % (len(keep), len(unique_tex) - len(keep)))

    # 5. Rebuild images/textures with only the kept canonical images.
    new_images = []
    new_textures = []
    remap_tex = {}
    for t in unique_tex:
        if t not in keep:
            continue
        src = textures[t]["source"]
        img = dict(images[src])
        remap_tex[t] = len(new_textures)
        new_images.append(img)
        nt = dict(textures[t])
        nt["source"] = len(new_images) - 1
        new_textures.append(nt)

    for mi, ti in enumerate(mat_tex):
        if ti is None:
            continue
        new_idx = remap_tex[canon[ti]]
        mats[mi]["pbrMetallicRoughness"]["baseColorTexture"]["index"] = new_idx

    j["images"] = new_images
    j["textures"] = new_textures

    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(j, f, separators=(",", ":"))

    return {img.get("uri") for img in new_images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gltf", nargs="+")
    ap.add_argument("--max-textures", type=int, default=64)
    args = ap.parse_args()

    keep_uris = set()
    gdirs = set()
    for gltf_path in args.gltf:
        keep_uris |= process(gltf_path, args.max_textures)
        gdirs.add(os.path.dirname(gltf_path))

    # 6. Delete PNGs not referenced by any processed glTF.
    removed_bytes = 0
    removed = 0
    for gdir in gdirs:
        for f in os.listdir(gdir):
            if f.lower().endswith(".png") and f not in keep_uris:
                p = os.path.join(gdir, f)
                removed_bytes += os.path.getsize(p)
                os.remove(p)
                removed += 1
    print("[fixup] removed %d orphaned PNGs (%.1f MB)"
          % (removed, removed_bytes / 1e6))


if __name__ == "__main__":
    main()
