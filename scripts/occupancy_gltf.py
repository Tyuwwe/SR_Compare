import json, struct, sys
import numpy as np

def trs(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4,4).T
    t = node.get("translation",[0,0,0]); r = node.get("rotation",[0,0,0,1]); s = node.get("scale",[1,1,1])
    x,y,z,w = r
    rm = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                   [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                   [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    m = np.eye(4)
    m[:3,:3] = rm * np.array(s)[None,:]
    m[:3,3] = t
    return m

CT = {5120:("b",1),5121:("B",1),5122:("h",2),5123:("H",2),5125:("I",4),5126:("f",4)}
NC = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}

def read_acc(j, buf, ai):
    a = j["accessors"][ai]
    bv = j["bufferViews"][a["bufferView"]]
    fmt, sz = CT[a["componentType"]]
    n = NC[a["type"]]
    off = bv.get("byteOffset",0) + a.get("byteOffset",0)
    stride = bv.get("byteStride", sz*n)
    count = a["count"]
    if stride == sz*n:
        arr = np.frombuffer(buf, dtype=np.dtype(fmt), count=count*n, offset=off).reshape(count,n)
    else:
        out = np.zeros((count,n), dtype=np.dtype(fmt))
        for i in range(count):
            o = off + i*stride
            out[i] = np.frombuffer(buf, dtype=np.dtype(fmt), count=n, offset=o)
        arr = out
    return arr

def main(path, ylo=1.2, yhi=2.2, res=0.5, x0=None, x1=None, z0=None, z1=None):
    j = json.load(open(path, encoding="utf-8"))
    import os
    binpath = os.path.join(os.path.dirname(path), j["buffers"][0]["uri"])
    buf = open(binpath,"rb").read()
    nodes = j["nodes"]
    tris = []  # world-space triangle vertices
    def walk(ni, parent):
        node = nodes[ni]
        m = parent @ trs(node)
        if "mesh" in node:
            for prim in j["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode",4) != 4: continue
                pos = read_acc(j, buf, prim["attributes"]["POSITION"]).astype(np.float64)
                if "indices" in prim:
                    idx = read_acc(j, buf, prim["indices"]).astype(np.int64).reshape(-1)
                else:
                    idx = np.arange(len(pos))
                p = pos[idx].reshape(-1,3,3)
                p = p @ m[:3,:3].T + m[:3,3]
                tris.append(p)
        for c in node.get("children",[]): walk(c,m)
    for ni in j["scenes"][j.get("scene",0)]["nodes"]:
        walk(ni, np.eye(4))
    T = np.concatenate(tris)
    print("triangles:", len(T))
    # keep triangles intersecting the eye-height band
    ymin = T[:,:,1].min(axis=1); ymax = T[:,:,1].max(axis=1)
    sel = (ymin <= yhi) & (ymax >= ylo)
    T = T[sel]
    print("in band:", len(T))
    X0, X1 = T[:,:,0].min(), T[:,:,0].max()
    Z0, Z1 = T[:,:,2].min(), T[:,:,2].max()
    if x0 is not None: X0, X1 = x0, x1
    if z0 is not None: Z0, Z1 = z0, z1
    print("x range %.1f..%.1f z range %.1f..%.1f" % (X0,X1,Z0,Z1))
    W = int((X1-X0)/res)+2; H = int((Z1-Z0)/res)+2
    grid = np.zeros((H,W), dtype=bool)
    for t in T:
        x0 = int((t[:,0].min()-X0)/res); x1 = int((t[:,0].max()-X0)/res)
        z0 = int((t[:,2].min()-Z0)/res); z1 = int((t[:,2].max()-Z0)/res)
        grid[z0:z1+1, x0:x1+1] = True
    for z in range(H-1,-1,-1):
        print("z=%6.1f " % (Z0+z*res) + "".join("#" if c else "." for c in grid[z]))
    print("        x from %.1f to %.1f, cell=%.2fm" % (X0, X1, res))

main(sys.argv[1],
     ylo=float(sys.argv[2]) if len(sys.argv) > 2 else 1.2,
     yhi=float(sys.argv[3]) if len(sys.argv) > 3 else 2.2,
     res=float(sys.argv[4]) if len(sys.argv) > 4 else 0.5,
     x0=float(sys.argv[5]) if len(sys.argv) > 5 else None,
     x1=float(sys.argv[6]) if len(sys.argv) > 6 else None,
     z0=float(sys.argv[7]) if len(sys.argv) > 7 else None,
     z1=float(sys.argv[8]) if len(sys.argv) > 8 else None)
