"""Apply/restore the audited PyTorch 2.11 Windows header compatibility patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PATCHES = [
    (
        'torch/csrc/dynamo/compiled_autograd.h',
        '9de6f875d6160f25ba9dc7ddc0126c0b167c4af8fea77bfefb0e9ce188311bc9',
        '398799bb95e6704840058073dec68f8145283ddf1576a21de5b9bcf3a35c9d77',
        b'    } else if constexpr (::std::is_same_v<T, ::std::string>) {\n      return at::StringType::get();',
        b'//     } else if constexpr (::std::is_same_v<T, ::std::string>) {\n//       return at::StringType::get();',
    ),
    (
        'c10/cuda/CUDACachingAllocator.h',
        '5c7a0e84ffd8f96f9ff8d5bc30dad64625e476d37a070631a4a0f8ef8d081be6',
        '550a5f63c6b11dadd3e214890f178eaea479a546e060fd5d349c0cc21e7cef6e',
        b'  StreamSegmentSize(cudaStream_t s, bool small, size_t sz)\n      : stream(s), is_small_pool(small), total_size(sz) {}',
        b'  StreamSegmentSize(cudaStream_t s, bool is_small_flag, size_t sz)\n      : stream(s), is_small_pool(is_small_flag), total_size(sz) {}',
    ),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    venv = ROOT / '.venv'
    if os.name != 'nt' or Path(sys.prefix).resolve() != venv.resolve():
        raise SystemExit('Run with this repository\'s .venv/Scripts/python.exe on Windows.')
    import torch
    if torch.__version__ != '2.11.0+cu128':
        raise SystemExit('Only the audited torch 2.11.0+cu128 build is supported.')
    if not Path(torch.__file__).resolve().is_relative_to(venv.resolve()):
        raise SystemExit('torch was imported from outside this repository venv.')
    setup = ROOT / 'environment-backups'
    operations = []
    # Validate every input before modifying either header.
    for relative, original_hash, patched_hash, old, new in PATCHES:
        header = Path(torch.__file__).parent / 'include' / relative
        data = header.read_bytes()
        digest = sha(data)
        if digest not in (original_hash, patched_hash):
            raise SystemExit(f'Unknown SHA256: {header}; refusing this version.')
        backup = setup / (header.name + '.original')
        if backup.exists() and sha(backup.read_bytes()) != original_hash:
            raise SystemExit(f'Unexpected backup: {backup}; refusing overwrite.')
        if args.restore:
            if digest == patched_hash and not backup.exists():
                raise SystemExit(f'Original backup is missing: {backup}')
            result = backup.read_bytes() if digest == patched_hash else data
        else:
            if b'\r\n' in data:
                old, new = old.replace(b'\n', b'\r\n'), new.replace(b'\n', b'\r\n')
            if digest == original_hash and data.count(old) != 1:
                raise SystemExit(f'Expected exactly one patch target: {header}')
            result = data.replace(old, new) if digest == original_hash else data
        expected = original_hash if args.restore else patched_hash
        if sha(result) != expected:
            raise SystemExit(f'Result does not match audited SHA256: {header}')
        operations.append((header, backup, data, result, original_hash))
    setup.mkdir(exist_ok=True)
    for header, backup, data, result, original_hash in operations:
        if not backup.exists() and sha(data) == original_hash:
            backup.write_bytes(data)
        if result != data:
            header.write_bytes(result)
        print(json.dumps({'header': str(header), 'changed': result != data,
                          'restoring': args.restore, 'backup': str(backup),
                          'sha256': sha(result)}, indent=2))


if __name__ == '__main__':
    main()
