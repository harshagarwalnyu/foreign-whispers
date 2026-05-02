#!/usr/bin/env python3
"""Verify pipeline artifacts for Foreign Whispers without Docker running."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

VIDEOS = [
    "Strait of Hormuz disruption threatens to shake global economy",
    "Alysa Liu: The 60 Minutes Interview",
    "Rob Reiner: The 60 Minutes Interview",
    "Military Drones: 60 Minutes Full Episodes",
]

STAGES = ["download", "captions", "transcribe", "translate", "tts", "dubbed_cap", "stitch"]


def ffprobe_duration(path: Path) -> float | None:
    """Return duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def check_vtt_monotonic(path: Path) -> list[str]:
    """Return list of violation descriptions if VTT timestamps are not monotonic."""
    violations = []
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})")

    def to_s(ts: str) -> float:
        h, m, s = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    prev_end = -1.0
    for i, m in enumerate(pattern.finditer(text)):
        start = to_s(m.group(1))
        end = to_s(m.group(2))
        if start < prev_end - 0.001:
            violations.append(f"  cue {i}: start {start:.3f}s < prev_end {prev_end:.3f}s")
        prev_end = end
    return violations


def kb(path: Path) -> str:
    return f"{path.stat().st_size / 1024:.1f}KB" if path.exists() else "—"


def check_json_segments(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return len(data.get("segments", []))
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Foreign Whispers pipeline artifacts")
    default_base = Path(__file__).parent.parent / "pipeline_data" / "api"
    parser.add_argument("--base", type=Path, default=default_base)
    args = parser.parse_args()
    base = args.base.resolve()

    print(f"\n{'='*70}")
    print(f"Foreign Whispers — Pipeline Artifact Verification")
    print(f"Base: {base}")
    print(f"{'='*70}\n")

    config_ids = sorted(
        d.name for d in (base / "dubbed_videos").iterdir()
        if d.is_dir()
    ) if (base / "dubbed_videos").exists() else []

    summary: dict[str, dict[str, str]] = {}

    for title in VIDEOS:
        print(f"{'─'*70}")
        print(f"VIDEO: {title}")
        print(f"{'─'*70}")
        row: dict[str, str] = {}

        # P1 — source video
        src_mp4 = base / "videos" / f"{title}.mp4"
        if src_mp4.exists():
            src_dur = ffprobe_duration(src_mp4)
            dur_str = f"{src_dur:.1f}s" if src_dur else "dur=?"
            print(f"  ✅ P1 video       {kb(src_mp4):>10}  {dur_str}")
            row["download"] = "✅"
        else:
            print(f"  ❌ P1 video       MISSING")
            row["download"] = "❌"
            src_dur = None

        # P1 — youtube captions
        cap_txt = base / "youtube_captions" / f"{title}.txt"
        if cap_txt.exists():
            print(f"  ✅ P1 captions    {kb(cap_txt):>10}")
            row["captions"] = "✅"
        else:
            print(f"  ❌ P1 captions    MISSING")
            row["captions"] = "❌"

        # P2 — transcription
        trans_json = base / "transcriptions" / "whisper" / f"{title}.json"
        seg_count = check_json_segments(trans_json)
        if trans_json.exists():
            print(f"  ✅ P2 transcribe  {kb(trans_json):>10}  {seg_count} segments")
            row["transcribe"] = "✅"
        else:
            print(f"  ❌ P2 transcribe  MISSING")
            row["transcribe"] = "❌"

        # P3 — translation
        transl_json = base / "translations" / "argos" / f"{title}.json"
        t_seg_count = check_json_segments(transl_json)
        if transl_json.exists():
            print(f"  ✅ P3 translate   {kb(transl_json):>10}  {t_seg_count} segments")
            row["translate"] = "✅"
        else:
            print(f"  ❌ P3 translate   MISSING")
            row["translate"] = "❌"

        # P4 — TTS wav files per config
        any_tts = False
        for cid in config_ids:
            tts_dir = base / "tts_audio" / "chatterbox" / cid
            wavs = list(tts_dir.glob(f"{title}*.wav")) if tts_dir.exists() else []
            if wavs:
                total_kb = sum(w.stat().st_size for w in wavs) / 1024
                print(f"  ✅ P4 tts/{cid}  {len(wavs)} wav  {total_kb:.0f}KB total")
                any_tts = True
        if not any_tts:
            print(f"  ❌ P4 tts         MISSING (no wav in any config)")
        row["tts"] = "✅" if any_tts else "❌"

        # P4 — dubbed captions VTT
        vtt = base / "dubbed_captions" / f"{title}.vtt"
        if vtt.exists():
            violations = check_vtt_monotonic(vtt)
            if violations:
                print(f"  ⚠️  P4 dubbed_cap {kb(vtt):>10}  timestamp violations:")
                for v in violations[:5]:
                    print(v)
                row["dubbed_cap"] = "⚠️"
            else:
                print(f"  ✅ P4 dubbed_cap {kb(vtt):>10}  timestamps OK")
                row["dubbed_cap"] = "✅"
        else:
            print(f"  ❌ P4 dubbed_cap  MISSING")
            row["dubbed_cap"] = "❌"

        # P5 — dubbed video per config
        any_stitch = False
        for cid in config_ids:
            dub_mp4 = base / "dubbed_videos" / cid / f"{title}.mp4"
            if dub_mp4.exists():
                dub_dur = ffprobe_duration(dub_mp4)
                drift_flag = ""
                if src_dur and dub_dur:
                    drift = abs(dub_dur - src_dur) / src_dur
                    drift_flag = f"  ⚠️ drift {drift:.0%}" if drift > 0.15 else f"  drift {drift:.0%}"
                dur_str = f"{dub_dur:.1f}s" if dub_dur else "dur=?"
                print(f"  ✅ P5 stitch/{cid}  {kb(dub_mp4):>10}  {dur_str}{drift_flag}")
                any_stitch = True
        if not any_stitch:
            print(f"  ❌ P5 stitch      MISSING (no dubbed mp4 in any config)")
        row["stitch"] = "✅" if any_stitch else "❌"

        summary[title[:40]] = row
        print()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    col_w = 11
    header = f"{'Video':<42}" + "".join(f"{s:^{col_w}}" for s in STAGES)
    print(header)
    print("─" * (42 + col_w * len(STAGES)))
    for title_key, row in summary.items():
        cells = "".join(f"{row.get(s, '?'):^{col_w}}" for s in STAGES)
        print(f"{title_key:<42}{cells}")

    missing = sum(1 for row in summary.values() for v in row.values() if v == "❌")
    warned = sum(1 for row in summary.values() for v in row.values() if v == "⚠️")
    print(f"\n{len(summary)} videos · {missing} missing · {warned} warnings")


if __name__ == "__main__":
    main()
