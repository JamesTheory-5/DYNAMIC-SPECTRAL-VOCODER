# DYNAMIC-SPECTRAL-VOCODER

```python
#!/usr/bin/env python3
"""
Hybrid Parametric Vocoder / Spectral Modeler
--------------------------------------------

Combines:
 - ESPRIT-based sinusoidal partial tracker
 - Residual / exciter modeling
 - Parametric synthesis with pitch, timbre, and noise control

Dependencies:
    numpy, scipy, librosa, soundfile
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import hilbert, savgol_filter, get_window
from partial_tracker_esprit import PartialTrackerESPRIT


# ================================================================
# === Analysis Utilities =========================================
# ================================================================

def frame_rms(y, frame_len, hop_len):
    rms = []
    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i + frame_len]
        rms.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    return np.array(rms)


def compute_noise_template(residual, sr, n_fft=2048):
    """Compute average magnitude spectrum as a noise 'timbre fingerprint'."""
    win = get_window("hann", n_fft, fftbins=True)
    if len(residual) < n_fft:
        residual = np.pad(residual, (0, n_fft - len(residual)))
    spec = np.fft.rfft(residual[:n_fft] * win, n=n_fft)
    mag = np.abs(spec)
    mag /= np.max(mag) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs, mag


def apply_noise_template(noise, sr, freqs, mag_template):
    """Shape white noise to match a stored spectral magnitude template."""
    n = len(noise)
    n_fft = (len(mag_template) - 1) * 2
    win = get_window("hann", n_fft, fftbins=True)
    out = np.zeros_like(noise)
    hop = n_fft // 4
    pos = 0
    while pos + n_fft <= n:
        white_frame = noise[pos:pos + n_fft] * win
        spec = np.fft.rfft(white_frame)
        spec *= mag_template
        frame_colored = np.fft.irfft(spec)
        out[pos:pos + n_fft] += frame_colored * win
        pos += hop
    return out / (np.max(np.abs(out)) + 1e-12)


# ================================================================
# === Core Analysis ==============================================
# ================================================================

def analyze_signal(y, sr):
    """
    Full analysis: sinusoidal + residual decomposition and parameter extraction.
    Returns a parametric analysis bundle.
    """
    tracker = PartialTrackerESPRIT(sr)
    tracks = tracker.track(y)
    y_harm = tracker.synthesize_from_signal(y, tracks)

    residual = y - y_harm
    analytic_env = np.abs(hilbert(residual))
    env_smooth = savgol_filter(analytic_env, 301, 3)

    rms_env = frame_rms(y, frame_len=int(0.02 * sr), hop_len=int(0.01 * sr))
    freqs_tmpl, mag_tmpl = compute_noise_template(residual, sr)

    bundle = {
        "sr": sr,
        "duration_s": len(y) / sr,
        "harmonic_tracks": tracks,
        "harmonic_render": y_harm,
        "exciter": {
            "residual_wave": residual,
            "noise_env": env_smooth,
            "noise_template_freqs": freqs_tmpl.tolist(),
            "noise_template_mag": mag_tmpl.tolist(),
        },
        "globals": {
            "rms_env": rms_env.tolist(),
        }
    }

    return bundle


# ================================================================
# === Synthesis & Control ========================================
# ================================================================

def pitch_shift_tracks(tracks, semitones):
    """Shift all harmonic frequencies by a semitone amount."""
    ratio = 2 ** (semitones / 12.0)
    new_tracks = {}
    for pid, tr in tracks.items():
        new_tracks[pid] = {
            "t": tr["t"][:],
            "freq": (np.array(tr["freq"]) * ratio).tolist(),
            "amp": tr["amp"][:],
            "phase0": tr["phase0"],
        }
    return new_tracks


def brighten_tracks(tracks, db_per_khz=6.0):
    """Boost amplitudes proportional to frequency (simple timbre brightener)."""
    new_tracks = {}
    for pid, tr in tracks.items():
        f = np.array(tr["freq"])
        boost_db = (f / 1000.0) * db_per_khz
        boost_lin = 10 ** (boost_db / 20.0)
        new_tracks[pid] = {
            "t": tr["t"][:],
            "freq": tr["freq"][:],
            "amp": (np.array(tr["amp"]) * boost_lin).tolist(),
            "phase0": tr["phase0"],
        }
    return new_tracks

from dynamic_spectral_env import analyze_dynamic_filter_sequence

def analyze_signal_with_dynamic_filter(y, sr):
    # 1. Run your original analysis_signal() logic
    bundle = analyze_signal(y, sr)

    # 2. Get exciter-ish drive signal and target residual
    residual = bundle["exciter"]["residual_wave"]
    # We'll use plain white noise as proxy exciter drive for fitting,
    # but there's an even better choice:
    # Use "flatish" residual estimate (residual / envelope) as x
    env = bundle["exciter"]["noise_env"]
    env_up = np.interp(np.arange(len(residual)), 
                       np.linspace(0, len(env)-1, len(residual)), 
                       env)
    safe_env = np.maximum(env_up, 1e-6)
    exciter_like = residual / safe_env  # try to flatten amplitude to isolate spectral behavior

    filter_frames, frame_meta = analyze_dynamic_filter_sequence(
        exciter_like,
        residual,
        fs=sr,
        frame_dur=0.03,
        hop_dur=0.01,
        max_order_a=6,
        max_order_b=6,
        reg=1e-6
    )

    bundle["exciter"]["dynamic_filter"] = {
        "filter_frames": filter_frames,
        "frame_meta": frame_meta,
    }

    return bundle

from dynamic_spectral_env import resynthesize_dynamic_filter_sequence, _apply_timevarying_filters

def resynthesize_exciter_layer(bundle, exciter_gain=1.0):
    sr = bundle["sr"]
    dur = bundle["duration_s"]

    exciter_info = bundle["exciter"]

    # Prepare optional global tilt (static coloration)
    # We'll recreate the same "noise_template_mag" shaping you already had.
    def static_color(noise_white):
        # reuse apply_noise_template-style shaping:
        # to keep it simple here, just return noise_white;
        # you can splice in your existing apply_noise_template if you want
        return noise_white

    # Pull dynamic filter info if it exists
    if "dynamic_filter" in exciter_info:
        filter_frames = exciter_info["dynamic_filter"]["filter_frames"]
        frame_meta    = exciter_info["dynamic_filter"]["frame_meta"]
    else:
        filter_frames = []
        frame_meta    = {
            "fs": sr,
            "frame_len": int(0.03 * sr),
            "hop_len": int(0.01 * sr),
        }

    noise_env = exciter_info["noise_env"]

    y_exc = resynthesize_dynamic_filter_sequence(
        fs=sr,
        duration_s=dur,
        filter_frames=filter_frames,
        frame_meta=frame_meta,
        exciter_gain=exciter_gain,
        noise_coloring=static_color,
        env_shape=noise_env,
    )

    return y_exc

def resynthesize_from_bundle(bundle,
                             pitch_fn=None,
                             amp_fn=None,
                             exciter_gain=1.0):

    sr = bundle["sr"]
    dur = bundle["duration_s"]
    tracks_in = bundle["harmonic_tracks"]

    # --- Transform harmonic tracks
    tracks_proc = tracks_in
    if pitch_fn is not None:
        tracks_proc = pitch_fn(tracks_proc)
    if amp_fn is not None:
        tracks_proc = amp_fn(tracks_proc)

    # --- Render harmonic layer ---
    tracker_dummy = PartialTrackerESPRIT(sr)
    y_harm = tracker_dummy.synthesize(tracks_proc, length=dur)

    # --- Render dynamic exciter layer (new version) ---
    y_exciter = resynthesize_exciter_layer(bundle, exciter_gain=exciter_gain)

    # --- Combine ---
    L = min(len(y_harm), len(y_exciter))
    y_out = y_harm[:L] + y_exciter[:L]
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-9)

    return y_out, y_harm[:L], y_exciter[:L]


# ================================================================
# === CLI / Example Usage ========================================
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Parametric Vocoder")
    parser.add_argument("filename", help="Input audio file")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="Pitch shift in semitones")
    parser.add_argument("--bright", type=float, default=0.0,
                        help="Brightness in dB per kHz")
    parser.add_argument("--exciter", type=float, default=1.0,
                        help="Exciter gain multiplier")
    parser.add_argument("--out", default="hybrid_vocode_out.wav",
                        help="Output filename")
    args = parser.parse_args()

    # === Load ===
    y, sr = librosa.load(args.filename, sr=None, mono=True)
    print(f"Analyzing {args.filename} ...")

    # === Analyze ===
    bundle = analyze_signal(y, sr)
    print("Analysis complete.")

    # === Build transform functions ===
    pitch_fn = (lambda tr: pitch_shift_tracks(tr, args.shift)) if args.shift != 0 else None
    amp_fn = (lambda tr: brighten_tracks(tr, args.bright)) if args.bright != 0 else None

    # === Resynthesize ===
    y_out, y_harm, y_exc = resynthesize_from_bundle(bundle,
                                                    pitch_fn=pitch_fn,
                                                    amp_fn=amp_fn,
                                                    exciter_gain=args.exciter)
    sf.write(args.out, y_out, sr)
    print(f"✅ Output saved to {args.out}")

```
