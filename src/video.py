"""
Video processing module for Uncensored Talking (@uncensored_talking).
Features:
- Dynamic Face Tracking & Centering via OpenCV Haar-Cascades.
- 3 Layout Modes: 'full_speaker' (1080x1920), 'full_broll' (1080x1920), and 'splitscreen' (top B-roll 1080x960, bottom speaker 1080x960).
- Seamless scene montage, master audio synchronization, and center-aligned ASS subtitle burning.
- Hardware-accelerated h264_nvenc rendering with CPU libx264 fallback.
"""

import logging
import os
import subprocess
from pathlib import Path

from .selector import ClipProposal, SceneCut, Segment

logger = logging.getLogger("uncensored_talking.video")

_NVENC_AVAILABLE: bool | None = None


def check_nvenc_support() -> bool:
    """
    Checks if FFmpeg supports h264_nvenc and whether an NVIDIA GPU is functional.
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    cmd = [
        "ffmpeg", "-v", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
        "-c:v", "h264_nvenc",
        "-f", "null", "-",
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            logger.info("Hardware acceleration verified: h264_nvenc is available and functional.")
            _NVENC_AVAILABLE = True
            return True
    except Exception:
        pass

    logger.info("NVENC not available. Falling back to CPU encoder (libx264).")
    _NVENC_AVAILABLE = False
    return False


def get_video_dimensions(video_path: Path | str) -> tuple[int, int]:
    """Returns (width, height) of the video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        w_str, h_str = res.stdout.strip().split(",")
        return int(w_str), int(h_str)
    except Exception:
        # Default standard 1080p 16:9
        return 1920, 1080


def detect_speaker_face_x(
    video_path: Path | str,
    start_time: float,
    end_time: float,
    orig_w: int,
    orig_h: int,
    crop_w: int,
) -> int:
    """
    Uses OpenCV Haar-Cascade to detect the speaker's face in the 16:9 original
    and calculates the optimal horizontal crop position (crop_x).
    Keeps the head in the horizontal center of the 9:16 frame.
    """
    default_x = max(0, (orig_w - crop_w) // 2)

    try:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return default_x

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        detected_centers: list[float] = []
        duration = max(0.5, end_time - start_time)
        # Sample up to 8 frames evenly spaced
        sample_times = [start_time + duration * (i + 1) / 9.0 for i in range(8)]

        for t in sample_times:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(50, 50),
            )
            if len(faces) > 0:
                # Pick largest face (main speaker)
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx = largest[0] + largest[2] / 2.0
                detected_centers.append(fx)

        cap.release()

        if detected_centers:
            # Use median center for stability against sudden head turns
            detected_centers.sort()
            median_face_x = detected_centers[len(detected_centers) // 2]
            crop_x = int(median_face_x - crop_w / 2.0)
            return max(0, min(crop_x, orig_w - crop_w))

    except Exception as exc:
        logger.debug(f"OpenCV face tracking fallback: {exc}")

    return default_x


def map_clip_time_to_source(t_local: float, segments: list[Segment]) -> float:
    """Translates local clip timeline (0.0 to dur) back into source video timestamp."""
    offset = 0.0
    for seg in segments:
        dur = seg.end - seg.start
        if t_local <= offset + dur + 0.001:
            return seg.start + max(0.0, t_local - offset)
        offset += dur
    return segments[-1].end


def escape_ffmpeg_filter_path(file_path: Path | str) -> str:
    """Escapes file paths for FFmpeg filter expressions."""
    path_str = Path(file_path).resolve().as_posix()
    return path_str.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def render_scene_plate(
    source_video: Path,
    scene: SceneCut,
    clip_segments: list[Segment],
    orig_w: int,
    orig_h: int,
    crop_w: int,
    output_plate: Path,
) -> Path:
    """
    Renders a single scene into an intermediate 1080x1920 30fps video plate
    according to its layout: 'full_speaker', 'full_broll', or 'splitscreen'.
    """
    output_plate.parent.mkdir(parents=True, exist_ok=True)
    dur = scene.duration
    src_start = map_clip_time_to_source(scene.start, clip_segments)
    src_end = map_clip_time_to_source(scene.end, clip_segments)

    # Face center position for the speaker in this scene
    crop_x = detect_speaker_face_x(
        video_path=source_video,
        start_time=src_start,
        end_time=src_end,
        orig_w=orig_w,
        orig_h=orig_h,
        crop_w=crop_w,
    )

    broll_path = Path(scene.broll_asset_path) if scene.broll_asset_path else None

    # Layout 1: full_speaker
    if scene.layout == "full_speaker" or not broll_path or not broll_path.exists():
        vf = f"crop={crop_w}:{orig_h}:{crop_x}:0,scale=1080:1920:flags=lanczos,setsar=1,fps=30"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{src_start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(source_video),
            "-vf", vf,
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output_plate),
        ]

    # Layout 2: full_broll
    elif scene.layout == "full_broll":
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(broll_path),
            "-vf", vf,
            "-t", f"{dur:.3f}",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output_plate),
        ]

    # Layout 3: splitscreen (Top B-Roll 1080x960, Bottom Speaker 1080x960)
    else:  # splitscreen
        filter_complex = (
            f"[0:v]scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960,setsar=1,fps=30[top];"
            f"[1:v]crop={crop_w}:{orig_h}:{crop_x}:0,scale=1080:960:flags=lanczos,setsar=1,fps=30[bot];"
            f"[top][bot]vstack=inputs=2[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(broll_path),
            "-ss", f"{src_start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(source_video),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-t", f"{dur:.3f}",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output_plate),
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_plate


def render_clip(
    source_video_path: Path | str,
    clip: ClipProposal,
    subtitles_ass_path: Path | str,
    output_dir: Path | str = Path("output"),
    force_encoder: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Renders the complete viral short:
    1. Renders each SceneCut according to layout ('full_speaker', 'full_broll', 'splitscreen')
       with face tracking.
    2. Stitches scene plates seamlessly together.
    3. Trims and matches the master audio track.
    4. Burns ASS subtitles centered on screen.
    5. Exports with h264_nvenc (or CPU fallback).
    """
    source_video_path = Path(source_video_path).resolve()
    subtitles_ass_path = Path(subtitles_ass_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = clip.title.strip().replace(" ", "_").lower()
    output_clip_path = output_dir / f"clip_{clip.clip_id}_{slug}.mp4"

    if output_clip_path.exists() and not overwrite:
        logger.info(f"Clip {clip.clip_id} already exists at: {output_clip_path}")
        return output_clip_path

    logger.info(
        f"Rendering Clip {clip.clip_id} ('{clip.title}'). Total duration: {clip.total_duration:.1f}s, "
        f"{len(clip.scenes)} scenes with dynamic layouts..."
    )

    if not clip.scenes:
        logger.info(f"No scenes provided for Clip {clip.clip_id}. Generating default full_speaker scene...")
        clip.scenes = [
            SceneCut(
                start=0.0,
                end=round(clip.total_duration, 2),
                layout="full_speaker",
                search_query="",
            )
        ]

    orig_w, orig_h = get_video_dimensions(source_video_path)
    # 9:16 crop width for the original height
    crop_w = int(orig_h * 9 / 16)
    if crop_w % 2 != 0:
        crop_w += 1

    temp_scenes_dir = Path("temp") / "scenes" / f"clip_{clip.clip_id}"
    temp_scenes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render each scene plate
    scene_plates: list[Path] = []
    for idx, scene in enumerate(clip.scenes, start=1):
        plate_file = temp_scenes_dir / f"scene_{idx}_{scene.layout}.mp4"
        logger.debug(
            f"  -> Scene {idx}/{len(clip.scenes)} [{scene.start:.1f}s-{scene.end:.1f}s]: "
            f"Layout={scene.layout}, Query='{scene.search_query}'"
        )
        rendered_plate = render_scene_plate(
            source_video=source_video_path,
            scene=scene,
            clip_segments=clip.segments,
            orig_w=orig_w,
            orig_h=orig_h,
            crop_w=crop_w,
            output_plate=plate_file,
        )
        scene_plates.append(rendered_plate)

    # 2. Concat list file
    concat_list_file = temp_scenes_dir / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for plate in scene_plates:
            f.write(f"file '{plate.resolve().as_posix()}'\n")

    # 3. Extract matching master audio for this clip
    clip_audio_file = temp_scenes_dir / "master_audio.wav"
    num_segs = len(clip.segments)
    if num_segs == 1:
        seg = clip.segments[0]
        cmd_audio = [
            "ffmpeg", "-y",
            "-ss", f"{seg.start:.3f}",
            "-t", f"{seg.end - seg.start:.3f}",
            "-i", str(source_video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(clip_audio_file),
        ]
    else:
        filter_parts = []
        for i, s in enumerate(clip.segments):
            filter_parts.append(
                f"[0:a]atrim=start={s.start:.3f}:end={s.end:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )
        concat_inputs = "".join(f"[a{i}]" for i in range(num_segs))
        filter_parts.append(f"{concat_inputs}concat=n={num_segs}:v=0:a=1[aout]")
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(source_video_path),
            "-filter_complex", ";".join(filter_parts),
            "-map", "[aout]",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(clip_audio_file),
        ]

    subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 4. Final assembly: Concatenated visual plates + Master Audio + Burn Subtitles
    if force_encoder:
        use_nvenc = force_encoder.lower() in ("nvenc", "h264_nvenc")
    else:
        env_pref = os.getenv("VIDEO_ENCODER", "auto").lower()
        if env_pref == "h264_nvenc":
            use_nvenc = True
        elif env_pref == "libx264":
            use_nvenc = False
        else:
            use_nvenc = check_nvenc_support()

    escaped_ass = escape_ffmpeg_filter_path(subtitles_ass_path)
    vf_subtitles = f"ass='{escaped_ass}'"

    cmd_final = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-i", str(clip_audio_file),
        "-vf", vf_subtitles,
        "-map", "0:v",
        "-map", "1:a",
    ]

    if use_nvenc:
        cmd_final.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p6",
            "-profile:v", "high",
            "-b:v", "8M",
            "-maxrate", "12M",
            "-bufsize", "16M",
            "-pix_fmt", "yuv420p",
        ])
    else:
        cmd_final.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ])

    cmd_final.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_clip_path),
    ])

    try:
        subprocess.run(cmd_final, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Successfully generated dynamic layout short: {output_clip_path}")
        return output_clip_path
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.strip() if exc.stderr else "Unknown FFmpeg error"
        logger.error(f"FFmpeg final rendering failed: {err_msg}")
        raise RuntimeError(f"FFmpeg final rendering failed: {err_msg}") from exc
