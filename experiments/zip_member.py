"""Read one member out of a zip file through a byte-range callback.

The NIH image archives are classic zip files under 4 GiB. A range callback
fetches the central directory and one compressed member, not the whole archive.
"""
from __future__ import annotations

import struct
import zlib

EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


def read_central_directory(fetch, file_size):
    """fetch(start, end) is inclusive. Returns members keyed by basename and zip path."""
    if file_size < 22:
        raise ValueError("zip is smaller than an end-of-central-directory record")
    tail = fetch(file_size - 22, file_size - 1)
    if not tail.startswith(EOCD) or tail[20:22] != b"\x00\x00":
        raise ValueError("zip comment or zip64 end record is not the locked NIH layout")
    cd_size, cd_offset = struct.unpack_from("<II", tail, 12)
    if cd_offset + cd_size + 22 != file_size:
        raise ValueError("central directory does not end at the zip tail")
    directory = fetch(cd_offset, cd_offset + cd_size - 1)
    members = {}
    cursor = 0
    while cursor < len(directory):
        if directory[cursor:cursor + 4] != CENTRAL:
            raise ValueError("central directory entry is not a file header")
        method, flags, crc, comp_size, uncomp_size, name_len, extra_len, comment_len, local_offset = _central_fields(
            directory, cursor,
        )
        name = directory[cursor + 46:cursor + 46 + name_len].decode("utf-8")
        members[name] = {
            "method": method,
            "flags": flags,
            "crc": crc,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "local_offset": local_offset,
        }
        cursor += 46 + name_len + extra_len + comment_len
    if cursor != len(directory):
        raise ValueError("central directory length does not match its entries")
    return members


def _central_fields(directory, cursor):
    flags, method = struct.unpack_from("<HH", directory, cursor + 8)
    crc, comp_size, uncomp_size = struct.unpack_from("<III", directory, cursor + 16)
    name_len, extra_len, comment_len = struct.unpack_from("<HHH", directory, cursor + 28)
    local_offset = struct.unpack_from("<I", directory, cursor + 42)[0]
    return method, flags, crc, comp_size, uncomp_size, name_len, extra_len, comment_len, local_offset


def extract_member(fetch, member):
    if member["flags"] & 0x1:
        raise ValueError("encrypted zip member")
    if member["method"] not in {0, 8}:
        raise ValueError(f"zip method {member['method']} is not stored or deflate")
    prefix = fetch(member["local_offset"], member["local_offset"] + 29)
    if prefix[:4] != LOCAL:
        raise ValueError("local file header is missing")
    name_len, extra_len = struct.unpack_from("<HH", prefix, 26)
    start = member["local_offset"] + 30 + name_len + extra_len
    compressed = fetch(start, start + member["comp_size"] - 1) if member["comp_size"] else b""
    if len(compressed) != member["comp_size"]:
        raise ValueError("compressed member was truncated")
    if member["method"] == 0:
        raw = compressed
    else:
        raw = zlib.decompress(compressed, -15)
    if len(raw) != member["uncomp_size"]:
        raise ValueError("uncompressed size does not match the central directory")
    if zlib.crc32(raw) & 0xFFFFFFFF != member["crc"]:
        raise ValueError("zip crc does not match")
    return raw
