#!/usr/bin/env python3
"""Read an uncompressed Blender 3.05 SDNA inventory without executing Blender.

This is a file inventory, not an evaluated-geometry or anatomical validator.
Python 3.11+, standard library only. Other blend versions fail explicitly.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def inventory(path):
    raw = path.read_bytes()
    if raw[:12] != b"BLENDER-v305":
        raise ValueError("Supported: uncompressed 64-bit little-endian Blender 3.05 only")
    blocks = []
    pos = 12
    while pos + 24 <= len(raw):
        code, size, address, schema, count = struct.unpack_from("<4sIQII", raw, pos)
        pos += 24
        if pos + size > len(raw):
            raise ValueError("Truncated block")
        blocks.append((code, address, schema, count, raw[pos:pos + size]))
        pos += size
        if code == b"ENDB":
            break
    dna = next(b[4] for b in blocks if b[0] == b"DNA1")
    pos = 0

    def marker(value):
        nonlocal pos
        if dna[pos:pos + 4] != value:
            raise ValueError(f"Missing SDNA marker {value!r}")
        pos += 4

    def number(fmt):
        nonlocal pos
        result = struct.unpack_from(fmt, dna, pos)
        pos += struct.calcsize(fmt)
        return result

    def strings():
        nonlocal pos
        result = []
        for _ in range(number("<I")[0]):
            end = dna.index(0, pos)
            result.append(dna[pos:end].decode())
            pos = end + 1
        pos = (pos + 3) & ~3
        return result

    marker(b"SDNA")
    marker(b"NAME")
    names = strings()
    marker(b"TYPE")
    types = strings()
    marker(b"TLEN")
    lengths = number("<" + "H" * len(types))
    pos = (pos + 3) & ~3
    marker(b"STRC")
    schemas = []
    for _ in range(number("<I")[0]):
        type_index, field_count = number("<HH")
        fields = {}
        offset = 0
        for _ in range(field_count):
            field_type, field_name = number("<HH")
            name = names[field_name]
            size = 8 if "*" in name else lengths[field_type]
            for dimension in re.findall(r"\[(\d+)\]", name):
                size *= int(dimension)
            fields[name.lstrip("*").split("[")[0]] = (offset, size)
            offset += size
        if offset != lengths[type_index]:
            raise ValueError(f"Unexpected SDNA alignment: {types[type_index]}")
        schemas.append((types[type_index], fields))
    definitions = dict(schemas)
    addresses = {b[1]: b for b in blocks}

    def field(blob, kind, name):
        start, size = definitions[kind][name]
        return blob[start:start + size]

    def integer(blob, kind, name):
        return int.from_bytes(field(blob, kind, name), "little")

    def id_name(blob, kind):
        value = field(field(blob, kind, "id"), "ID", "name")
        return value.split(b"\0")[0].decode(errors="replace")[2:]

    objects = []
    for code, _, _, _, blob in blocks:
        if code != b"OB\0\0":
            continue
        type_code = integer(blob, "Object", "type")
        row = {"name": id_name(blob, "Object"), "type_code": type_code,
               "type": {1: "MESH", 2: "CURVE", 4: "SURFACE"}.get(type_code, "OTHER")}
        target = addresses.get(integer(blob, "Object", "data"))
        if target:
            kind = schemas[target[2]][0]
            if kind == "Mesh":
                row.update(vertices=integer(target[4], kind, "totvert"),
                           polygons=integer(target[4], kind, "totpoly"))
            elif kind == "Curve":
                # Nonempty spline list is stronger evidence than an object name.
                nurb = field(target[4], kind, "nurb")
                row["has_splines"] = bool(integer(nurb, "ListBase", "first"))
        objects.append(row)
    return {"input": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "blend_version": "3.05", "method": "read_only_sdna_not_evaluated_geometry",
            "fma_properties_inspected": False, "object_count": len(objects), "objects": objects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.blend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Inventoried {result['object_count']} objects: {args.output}")


if __name__ == "__main__":
    main()
