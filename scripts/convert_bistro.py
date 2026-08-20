"""Blender CLI script: convert Amazon Lumberyard Bistro FBX to glTF.

Usage:
  blender.exe --background --factory-startup --python scripts/convert_bistro.py -- \
      --input <fbx> [<fbx> ...] --output <dir> --name <basename>

Multiple inputs are imported into one scene and exported as a single glTF.
Exports GLTF_SEPARATE (.gltf + .bin + textures). DDS textures are converted
to PNG by the exporter (export_image_format='AUTO').
"""

import argparse
import os
import sys

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Clean default scene (factory startup gives us an empty scene, but be safe).
    bpy.ops.wm.read_factory_settings(use_empty=True)

    for fbx in args.input:
        print("[convert_bistro] importing %s" % fbx)
        # Native C++ FBX importer (the legacy Python addon io_scene_fbx is
        # broken on Blender 5.1 for files containing lights).
        bpy.ops.wm.fbx_import(filepath=fbx)

    out_path = os.path.join(args.output, args.name + ".gltf")
    print("[convert_bistro] exporting %s" % out_path)
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format="GLTF_SEPARATE",
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_yup=True,
        export_apply=False,
        export_animations=False,
    )
    print("[convert_bistro] done")


if __name__ == "__main__":
    main()
