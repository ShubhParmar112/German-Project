# metal_app/views.py
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .audio_processor import predict_metal_from_file
import os
import uuid
import librosa
import numpy as np
import json


def home(request):
    """Renders the dashboard template (no processing on GET)."""
    return render(request, 'index.html')


@csrf_exempt
def analyze_api(request):
    """
    API endpoint: accepts audio file via POST, returns JSON with
    prediction + all analysis data for interactive charts.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    audio_file = request.FILES.get('audio_file')
    if not audio_file:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    # Save uploaded audio to temp location
    unique_id = str(uuid.uuid4())[:8]
    file_name = f"recording_{unique_id}.wav"
    file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb+') as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    # Run prediction
    prediction = predict_metal_from_file(file_path)

    # Generate analysis data
    analysis = generate_analysis_data(file_path)

    # Cleanup
    try:
        os.remove(file_path)
    except OSError:
        pass

    return JsonResponse({
        'prediction': prediction,
        'analysis': analysis,
    })


def generate_analysis_data(audio_path):
    """
    Load audio and compute all analysis data, returning numerical arrays
    for the frontend to render with Chart.js.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr

    # Decimate for browser performance - max ~2000 points per chart
    max_points = 2000
    step = max(1, len(y) // max_points)
    y_dec = y[::step]
    times_dec = np.linspace(0, duration, len(y_dec))

    # ---- 1. Waveform ----
    waveform = {
        'times': times_dec.tolist(),
        'amplitudes': y_dec.tolist(),
    }

    # ---- 2. Spectrogram ----
    # Compute STFT with reasonable resolution
    n_fft = 1024
    hop_length = 256
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    # Decimate spectrogram: keep ~200 time bins and ~128 freq bins
    freq_step = max(1, D_db.shape[0] // 128)
    time_step = max(1, D_db.shape[1] // 200)
    D_dec = D_db[::freq_step, ::time_step]

    spec_times = np.linspace(0, duration, D_dec.shape[1])
    spec_freqs = np.linspace(0, sr / 2, D_dec.shape[0])

    spectrogram = {
        'times': spec_times.tolist(),
        'frequencies': spec_freqs.tolist(),
        'magnitudes': D_dec.tolist(),
    }

    # ---- 3. RMS Energy ----
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    rms_step = max(1, len(rms) // max_points)

    rms_energy = {
        'times': rms_times[::rms_step].tolist(),
        'values': rms[::rms_step].tolist(),
    }

    # ---- 4. FFT Spectrum ----
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    fft_db = 20 * np.log10(fft[1:] + 1e-8)
    freqs_pos = freqs[1:]
    fft_step = max(1, len(freqs_pos) // max_points)

    fft_spectrum = {
        'frequencies': freqs_pos[::fft_step].tolist(),
        'magnitudes': fft_db[::fft_step].tolist(),
    }

    # ---- 5. Amplitude Distribution (Histogram) ----
    counts, bin_edges = np.histogram(y, bins=100)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2)

    histogram = {
        'bins': bin_centers.tolist(),
        'counts': counts.tolist(),
    }

    # ---- 6. Moving RMS Envelope ----
    envelope = np.abs(y)
    window = 1000
    moving_rms_arr = np.convolve(envelope, np.ones(window) / window, mode='valid')
    mrms_times = np.linspace(0, duration, len(moving_rms_arr))
    mrms_step = max(1, len(moving_rms_arr) // max_points)

    moving_rms = {
        'times': mrms_times[::mrms_step].tolist(),
        'values': moving_rms_arr[::mrms_step].tolist(),
    }

    # ---- Audio info ----
    audio_info = {
        'sample_rate': int(sr),
        'duration': round(float(duration), 3),
        'peak_amplitude': round(float(np.max(np.abs(y))), 4),
        'num_samples': len(y),
    }

    return {
        'waveform': waveform,
        'spectrogram': spectrogram,
        'rms_energy': rms_energy,
        'fft_spectrum': fft_spectrum,
        'histogram': histogram,
        'moving_rms': moving_rms,
        'audio_info': audio_info,
    }