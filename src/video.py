"""
Video processing module for Uncensored Talking (@uncensored_talking).
Anti-Stretch & Dynamic Face Tracking:
- STRICT PROPORTIONAL SCALING: scale=-1:1920 (full) or scale=-1:960 (split) followed by crop.
  Never stretches video.
- DYNAMIC CENTERING: Tracks face/forehead center using MediaPipe / OpenCV with a smoothed
  moving average over 30 frames.
- 3 Layout Modes: 'full_speaker', 'full_broll', 'splitscreen'.
- Frame-accurate subtitle overlay burning using Pillow transparent PNG sequence with vertical gradient.
- Hardware-accelerated h264_nvenc rendering with CPU libx264 fallback.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .selector import ClipProposal, SceneCut, Segment

logger = logging.getLogger("uncensored_talking.video")

_NVENC_AVAILABLE: bool | None = None


def check_nvenc_support() -> bool:
    """Checks if FFmpeg supports h264_nvenc and hardware encoder is functional."""
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
        return 1920, 1080


def escape_ffmpeg_filter_path(file_path: Path | str) -> str:
    """Escapes file paths for FFmpeg filter expressions."""
    path_str = Path(file_path).resolve().as_posix()
    return path_str.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def compute_smoothed_face_trajectory(
    video_path: Path | str,
    start_time: float,
    end_time: float,
    orig_w: int,
    orig_h: int,
    target_crop_w: int,
    target_h: int,
    smooth_window: int = 30,
) -> int:
    """
    Tracks face/forehead center using MediaPipe / OpenCV over the scene duration.
    Calculates moving average over 30 frames to guarantee smooth horizontal centering
    without sudden jumps or jitter.
    Returns the smoothed crop_x coordinate in proportionally scaled space.
    """
    scale_factor = target_h / float(orig_h)
    scaled_w = int(orig_w * scale_factor)
    default_crop_x = max(0, (scaled_w - target_crop_w) // 2)

    duration = max(0.4, end_time - start_time)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_crop_x

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    num_frames = max(1, end_frame - start_frame)

    # Sample 30 frames evenly across the scene
    sample_indices = np.linspace(start_frame, max(start_frame, end_frame - 1), min(30, num_frames), dtype=int)

    # 1. Try MediaPipe Face Detection
    mp_face_detector = None
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            mp_face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
    except Exception:
        mp_face_detector = None

    # OpenCV Cascade Fallback
    cascade = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass

    raw_centers: list[float] = []

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        center_x = None

        # MediaPipe attempt
        if mp_face_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_detector.process(rgb)
            if results and results.detections:
                # Primary face
                det = results.detections[0]
                bb = det.location_data.relative_bounding_box
                # Forehead / eye level center
                center_x = (bb.xmin + bb.width / 2.0) * orig_w

        # OpenCV Haar fallback
        if center_x is None and cascade is not None and not cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))
            if len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                center_x = largest[0] + largest[2] / 2.0

        if center_x is not None:
            raw_centers.append(center_x)

    cap.release()
    if mp_face_detector is not None:
        try:
            mp_face_detector.close()
        except Exception:
            pass

    if not raw_centers:
        return default_crop_x

    # 2. Compute 30-frame moving average smoothing
    if len(raw_centers) >= 5:
        w_size = min(smooth_window, len(raw_centers))
        kernel = np.ones(w_size) / w_size
        smoothed = np.convolve(raw_centers, kernel, mode="valid")
        final_center = float(np.mean(smoothed))
    else:
        final_center = float(np.median(raw_centers))

    # Translate to proportionally scaled space
    scaled_center_x = final_center * scale_factor
    crop_x = int(scaled_center_x - target_crop_w / 2.0)
    return max(0, min(crop_x, scaled_w - target_crop_w))


def map_clip_time_to_source(t_local: float, segments: list[Segment]) -> float:
    """Translates local clip timeline back to source video timestamp."""
    offset = 0.0
    for seg in segments:
        dur = seg.end - seg.start
        if t_local <= offset + dur + 0.001:
            return seg.start + max(0.0, t_local - offset)
        offset += dur
    return segments[-1].end


def render_scene_plate(
    source_video: Path,
    scene: SceneCut,
    clip_segments: list[Segment],
    orig_w: int,
    orig_h: int,
    output_plate: Path,
) -> Path:
    """
    Renders a single scene plate:
    - full_speaker: 1080x1920 (proportional scale=-1:1920, smoothed face crop).
    - full_broll: 1080x1920 (proportional scale=-1:1920, center crop).
    - splitscreen: Top B-Roll 1080x960, Bottom Speaker 1080x960 (proportional scale=-1:960, smoothed face crop).
    """
    output_plate.parent.mkdir(parents=True, exist_ok=True)
    dur = scene.duration
    src_start = map_clip_time_to_source(scene.start, clip_segments)
    src_end = map_clip_time_to_source(scene.end, clip_segments)

    broll_path = Path(scene.broll_asset_path) if scene.broll_asset_path else None

    # 1. full_speaker (1080x1920)
    if scene.layout == "full_speaker" or not broll_path or not broll_path.exists():
        crop_x = compute_smoothed_face_trajectory(
            video_path=source_video,
            start_time=src_start,
            end_time=src_end,
            orig_w=orig_w,
            orig_h=orig_h,
            target_crop_w=1080,
            target_h=1920,
        )
        vf = (
            f"scale=-1:1920:flags=lanczos,"
            f"crop=1080:1920:{crop_x}:0,"
            f"setsar=1,fps=30"
        )
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

    # 2. full_broll (1080x1920)
    elif scene.layout == "full_broll":
        vf = (
            "scale=-1:1920:flags=lanczos,"
            "crop=1080:1920:(in_w-1080)/2:0,"
            "setsar=1,fps=30"
        )
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

    # 3. splitscreen (Top B-Roll 1080x960, Bottom Speaker 1080x960)
    else:  # splitscreen
        crop_x_split = compute_smoothed_face_trajectory(
            video_path=source_video,
            start_time=src_start,
            end_time=src_end,
            orig_w=orig_w,
            orig_h=orig_h,
            target_crop_w=1080,
            target_h=960,
        )
        filter_complex = (
            f"[0:v]scale=-1:960:flags=lanczos,crop=1080:960:(in_w-1080)/2:0,setsar=1,fps=30[top];"
            f"[1:v]scale=-1:960:flags=lanczos,crop=1080:960:{crop_x_split}:0,setsar=1,fps=30[bot];"
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


DEFAULT_BG_MUSIC_PATH = (Path(__file__).parent.parent / "assets" / "audio" / "background_music.mp3").resolve()


def render_clip(
    source_video_path: Path | str,
    clip: ClipProposal,
    subtitles_source: Path | str | None = None,
    output_dir: Path | str = Path("output"),
    force_encoder: str | None = None,
    overwrite: bool = True,
    subtitles_ass_path: Path | str | None = None,
    bg_music_path: Path | str | None = DEFAULT_BG_MUSIC_PATH,
    bg_music_volume: float = 0.12,
    output_filename: str | Path | None = None,
) -> Path:
    """
    Renders complete viral short:
    1. Renders scene cuts with anti-stretch proportional scaling & dynamic face tracking.
    2. Stitches visual scene plates together.
    3. Extracts master audio corresponding to non-linear cut segments.
    4. Mixes pre-cut background music track ('Me and the Devil' from s32) under speech.
    5. Burns in Pillow PNG gradient subtitles via FFmpeg overlay (or ASS fallback).
    6. Exports with NVENC or libx264.
    """
    source_video_path = Path(source_video_path).resolve()
    sub_arg = subtitles_source or subtitles_ass_path
    if sub_arg is None:
        raise ValueError("Either subtitles_source or subtitles_ass_path must be provided to render_clip.")
    subtitles_path = Path(sub_arg).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        out_name = Path(output_filename).name
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"
        output_clip_path = output_dir / out_name
    elif clip.title.startswith("story_"):
        output_clip_path = output_dir / f"{clip.title}.mp4"
    else:
        slug = clip.title.strip().replace(" ", "_").lower()
        output_clip_path = output_dir / f"clip_{clip.clip_id}_{slug}.mp4"

    if output_clip_path.exists() and not overwrite:
        logger.info(f"Clip {clip.clip_id} already exists at: {output_clip_path}")
        return output_clip_path

    if not clip.scenes:
        clip.scenes = [
            SceneCut(
                start=0.0,
                end=round(clip.total_duration, 2),
                layout="full_speaker",
                search_query="",
            )
        ]

    orig_w, orig_h = get_video_dimensions(source_video_path)
    logger.info(
        f"Rendering Clip {clip.clip_id} ('{clip.title}'). Source: {orig_w}x{orig_h}, "
        f"Duration: {clip.total_duration:.1f}s across {len(clip.scenes)} scenes..."
    )

    temp_scenes_dir = Path("temp") / "scenes" / f"clip_{clip.clip_id}"
    temp_scenes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render each scene plate
    scene_plates: list[Path] = []
    for idx, scene in enumerate(clip.scenes, start=1):
        plate_file = temp_scenes_dir / f"scene_{idx}_{scene.layout}.mp4"
        rendered_plate = render_scene_plate(
            source_video=source_video_path,
            scene=scene,
            clip_segments=clip.segments,
            orig_w=orig_w,
            orig_h=orig_h,
            output_plate=plate_file,
        )
        scene_plates.append(rendered_plate)

    # 2. Concat list file
    concat_list_file = temp_scenes_dir / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for plate in scene_plates:
            f.write(f"file '{plate.resolve().as_posix()}'\n")

    # 3. Master audio track extraction
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

    # 4. Final render with subtitle overlay
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

    active_bg_music: Path | None = None
    if bg_music_path:
        bmp = Path(bg_music_path).resolve()
        if bmp.is_file():
            active_bg_music = bmp
            logger.info(f"Embedding background music: {active_bg_music.name} (volume: {bg_music_volume:.2f})")
        else:
            logger.warning(f"Background music file not found at {bmp}. Proceeding without background music.")

    # Determine if subtitle source is PNG concat file or ASS file
    is_png_concat = subtitles_path.name.endswith(".txt") or (
        subtitles_path.is_file() and "subtitles_concat" in subtitles_path.name
    )

    if is_png_concat:
        # Burning via FFmpeg overlay
        if active_bg_music:
            filter_complex = [
                "[0:v][1:v]overlay=0:0:shortest=1[v_final]",
                "[2:a]volume=1.0,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a_speech]",
                f"[3:a]volume={bg_music_volume:.2f},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a_bg]",
                "[a_speech][a_bg]amix=inputs=2:duration=first:dropout_transition=2[a_final]",
            ]
            cmd_final = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
                "-f", "concat", "-safe", "0", "-i", str(subtitles_path),
                "-i", str(clip_audio_file),
                "-stream_loop", "-1", "-i", str(active_bg_music),
                "-filter_complex", ";".join(filter_complex),
                "-map", "[v_final]",
                "-map", "[a_final]",
            ]
        else:
            cmd_final = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
                "-f", "concat", "-safe", "0", "-i", str(subtitles_path),
                "-i", str(clip_audio_file),
                "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v_final]",
                "-map", "[v_final]",
                "-map", "2:a",
            ]
    else:
        # Fallback ASS burning
        escaped_ass = subtitles_path.as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        if active_bg_music:
            filter_complex = [
                f"[0:v]ass='{escaped_ass}'[v_final]",
                "[1:a]volume=1.0,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a_speech]",
                f"[2:a]volume={bg_music_volume:.2f},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a_bg]",
                "[a_speech][a_bg]amix=inputs=2:duration=first:dropout_transition=2[a_final]",
            ]
            cmd_final = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
                "-i", str(clip_audio_file),
                "-stream_loop", "-1", "-i", str(active_bg_music),
                "-filter_complex", ";".join(filter_complex),
                "-map", "[v_final]",
                "-map", "[a_final]",
            ]
        else:
            cmd_final = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
                "-i", str(clip_audio_file),
                "-vf", f"ass='{escaped_ass}'",
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
