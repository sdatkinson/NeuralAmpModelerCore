#!/usr/bin/env python3
"""Compare two builds of tools/render on the same models and audio, for reviewing a speed-up:

    python3 tools/compare_renders.py build-base/tools/render build-candidate/tools/render
    python3 tools/compare_renders.py BASE CANDIDATE --models example_models/A2.nam my_ir.wav

Every model (default: example_models/*.nam) is rendered by both binaries and reported as EXACT, NEAR (error-to-signal
ratio below -100 dB and no sample off by more than 1e-4) or DIFFERENT, with the first differing sample. Look at speed numbers only when every model is
EXACT or NEAR. Exits 0 if none is DIFFERENT, 1 if one is, 2 if a render fails. Standard library only;
--self-test checks the comparison itself.

render writes output channel 0 only, so only that channel is compared.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

# Float32 differences between code paths measure -117 to -140 dB; real bugs are tens of dB.
NEAR_DB = -100.0
# ...and no single sample may differ by more than this (full scale is 1.0). Float32 path differences measured here
# stay below 2e-5; a whole-file ratio alone would let one wrong sample in a long render through.
NEAR_MAX_ABS = 1e-4


def read_float_wav(path):
    blob = path.read_bytes()
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path}: expected a RIFF/WAVE file")
    fmt = audio = None
    pos = 12
    while pos + 8 <= len(blob):
        kind, size = struct.unpack_from("<4sI", blob, pos)
        pos += 8
        if pos + size > len(blob):
            raise ValueError(f"{path}: truncated WAV chunk")
        if kind == b"fmt ":
            fmt = blob[pos:pos + size]
        elif kind == b"data":
            audio = blob[pos:pos + size]
        pos += size + (size & 1)
    if fmt is None or audio is None or len(fmt) < 16:
        raise ValueError(f"{path}: missing WAV format or audio data")
    tag, channels, rate, _, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 65534 and len(fmt) >= 40:
        if fmt[24:40] == bytes.fromhex("0300000000001000800000aa00389b71"):
            tag = 3  # WAVE_FORMAT_EXTENSIBLE with IEEE float subtype
    if (tag, channels, bits, align) != (3, 1, 32, 4) or rate == 0:
        raise ValueError(f"{path}: unsupported WAV format (need mono 32-bit IEEE float)")
    if len(audio) % 4:
        raise ValueError(f"{path}: incomplete float sample")
    samples = [item[0] for item in struct.iter_unpack("<f", audio)]
    return rate, audio, samples


def input_frames(path):
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes()
    except wave.Error:  # the wave module reads integer PCM only
        return len(read_float_wav(path)[2])


def compare(base_path, candidate_path, expected_frames=None):
    base_rate, base_bytes, base = read_float_wav(base_path)
    candidate_rate, candidate_bytes, candidate = read_float_wav(candidate_path)
    result = {"verdict": "DIFFERENT", "identical": False,
              "max_abs_diff": None, "first_differing_sample": None, "esr_db": None}
    if base_rate != candidate_rate or len(base) != len(candidate):
        result["reason"] = (f"sample rate/count: {base_rate} Hz/{len(base)} vs "
                            f"{candidate_rate} Hz/{len(candidate)}")
        return result
    # Two renders that are equally empty or equally truncated agree with each other, and are both wrong.
    if expected_frames is not None and len(base) != expected_frames:
        result["reason"] = f"both renders have {len(base)} samples; the input has {expected_frames}"
        return result
    for name, samples in (("base", base), ("candidate", candidate)):
        bad = next((i for i, v in enumerate(samples) if not math.isfinite(v)), None)
        if bad is not None:
            result.update(first_differing_sample=bad, reason=f"{name} render has a non-finite sample at {bad}")
            return result
    if base_bytes == candidate_bytes:
        result.update(verdict="EXACT", identical=True, max_abs_diff=0.0)
        return result
    first = next(i for i in range(len(base))
                 if base_bytes[4*i:4*i+4] != candidate_bytes[4*i:4*i+4])
    diffs = [a - b for a, b in zip(base, candidate)]
    error = math.fsum(d * d for d in diffs)
    energy = math.fsum(a * a for a in base)
    esr = 10 * math.log10(error / energy) if error and energy else (-math.inf if not error else math.inf)
    max_abs = max(abs(d) for d in diffs)
    result.update(verdict="NEAR" if esr <= NEAR_DB and max_abs <= NEAR_MAX_ABS else "DIFFERENT",
                  max_abs_diff=max_abs,
                  first_differing_sample=first,
                  esr_db=esr if math.isfinite(esr) else ("-Infinity" if esr < 0 else "Infinity"))
    return result


def wav_rate(path):
    """Sample rate from a WAV's fmt chunk, whatever its sample format."""
    blob = path.read_bytes()
    pos = 12
    while pos + 8 <= len(blob):
        kind, size = struct.unpack_from("<4sI", blob, pos)
        if kind == b"fmt " and size >= 8:
            return struct.unpack_from("<I", blob, pos + 12)[0]
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path}: no WAV format chunk")


def model_sample_rate(model):
    """The rate to generate the test input at: the model's own rate, or 48 kHz when it does not declare one
    (render then runs the model at the input's rate). A .wav model (an impulse response) runs at its file's rate."""
    if model.suffix.lower() == ".wav":
        return wav_rate(model)
    rate = json.loads(model.read_text(encoding="utf-8")).get("sample_rate")
    return int(rate) if rate and rate > 0 else 48000


def write_test_input(path, rate):
    samples = []
    duration = 0.5
    slope = math.log(20000 / 20)
    for i in range(int(rate * duration)):
        t = i / rate
        phase = 2 * math.pi * 20 * duration * math.expm1(slope * t / duration) / slope
        samples.append(int(round(0.8 * math.sin(phase) * 32767)))
    seed = 0x12345678
    for _ in range(rate // 2):
        seed = (1664525 * seed + 1013904223) & 0xffffffff
        samples.append(int(((seed / 0xffffffff) * 2 - 1) * 16384))
    samples.extend([0] * (rate // 4))
    samples.extend(32767 if (i // 24) % 2 == 0 else -32768 for i in range(rate // 4))
    # An odd number of extra samples: no power-of-two block size above 1 divides the length, so the last block is short
    samples.extend(int(16384 * math.sin(0.05 * i)) for i in range(37))
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        out.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def write_float_wav(path, samples, rate=48000):
    audio = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, 4 * rate, 4, 32)
    path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(audio)) + b"WAVE"
                     + b"fmt " + struct.pack("<I", 16) + fmt
                     + b"data" + struct.pack("<I", len(audio)) + audio)


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        base, candidate = (Path(directory) / name for name in ("base.wav", "candidate.wav"))
        write_float_wav(base, [1.0, 0.0])
        for value, verdict, expected_esr in ((1.0, "EXACT", None),
                                             (1.0 + 2**-23, "NEAR", -138.473798),
                                             (1.001, "DIFFERENT", -60.0)):
            write_float_wav(candidate, [value, 0.0])
            result = compare(base, candidate)
            if result["verdict"] != verdict:
                raise AssertionError(f"expected {verdict}: {result}")
            if expected_esr is not None:
                if result["first_differing_sample"] != 0 or abs(result["esr_db"] - expected_esr) > 0.01:
                    raise AssertionError(f"incorrect ESR or first sample: {result}")
        # One wrong sample in a long render, and a NaN, are DIFFERENT (the whole-file ratio alone would say NEAR).
        long_base = [math.sin(0.01 * i) for i in range(60037)]
        write_float_wav(base, long_base)
        for i, bad in ((30000, long_base[30000] + 0.002), (17, float("nan"))):
            write_float_wav(candidate, long_base[:i] + [bad] + long_base[i + 1:])
            result = compare(base, candidate)
            if result["verdict"] != "DIFFERENT" or result["first_differing_sample"] != i:
                raise AssertionError(f"one sample changed at {i} to {bad}: {result}")
        # Equally empty or truncated renders are not EXACT.
        write_float_wav(candidate, [1.0, 0.0])
        for frames in ([], [1.0]):
            write_float_wav(base, frames)
            write_float_wav(candidate, frames)
            result = compare(base, candidate, expected_frames=2)
            if result["verdict"] != "DIFFERENT":
                raise AssertionError(f"{len(frames)}-sample renders of a 2-sample input: {result}")
        # The generated input follows the model's rate; render refuses a mismatched one.
        model = Path(directory) / "model.nam"
        for written, expected in (({"sample_rate": 44100}, 44100), ({"sample_rate": 48000.0}, 48000), ({}, 48000),
                                  ({"sample_rate": -1}, 48000)):
            model.write_text(json.dumps(written))
            rate = model_sample_rate(model)
            write_test_input(Path(directory) / "input.wav", rate)
            with wave.open(str(Path(directory) / "input.wav"), "rb") as wav:
                if (rate, wav.getframerate()) != (expected, expected):
                    raise AssertionError(f"{written}: model rate {rate}, input rate {wav.getframerate()}")
                if wav.getnframes() % 2 == 0:
                    raise AssertionError(f"input length {wav.getnframes()} is even; the last block would not be short")
        ir = Path(directory) / "ir.wav"
        write_float_wav(ir, [1.0, 0.5], rate=44100)
        if model_sample_rate(ir) != 44100:
            raise AssertionError(f"a .wav model's rate: {model_sample_rate(ir)}")


def main():
    parser = argparse.ArgumentParser(description="Compare audio from two render builds")
    parser.add_argument("base_render_bin", nargs="?", type=Path)
    parser.add_argument("candidate_render_bin", nargs="?", type=Path)
    parser.add_argument("--models", nargs="+", type=Path,
                        default=sorted((Path(__file__).resolve().parent.parent / "example_models").glob("*.nam")),
                        help="model files (.nam, or .wav impulse responses); default: example_models/*.nam")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.base_render_bin or not args.candidate_render_bin:
        parser.error("BASE_RENDER_BIN and CANDIDATE_RENDER_BIN are required")
    if not args.models:
        parser.error("no models found; pass --models")
    results = []
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        for index, model in enumerate(args.models):
            if args.input:
                input_path = args.input.resolve()
            else:
                # The generated input is at the model's own rate, so a 44.1 kHz model is not fed 48 kHz audio.
                rate = model_sample_rate(model)
                input_path = temp / f"input-{rate}.wav"
                if not input_path.exists():
                    write_test_input(input_path, rate)
            outputs = [temp / f"{index}-base.wav", temp / f"{index}-candidate.wav"]
            for binary, output in zip((args.base_render_bin, args.candidate_render_bin), outputs):
                run = subprocess.run([str(binary.resolve()), str(model.resolve()),
                                      str(input_path), str(output)], capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError(f"{binary} failed on {model}: {run.stderr.strip() or run.stdout.strip()}")
            results.append({"model": str(model), **compare(*outputs, input_frames(input_path))})
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            line = f"{result['model']}: {result['verdict']}"
            if result.get("reason"):
                line += f" ({result['reason']})"
            elif not result["identical"]:
                line += (f" max_abs_diff={result['max_abs_diff']:.9g}"
                         f" first_differing_sample={result['first_differing_sample']}"
                         f" ESR={result['esr_db']} dB")
            print(line)
    return int(any(result["verdict"] == "DIFFERENT" for result in results))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, wave.Error) as error:
        print(f"compare_renders: {error}", file=sys.stderr)
        sys.exit(2)
