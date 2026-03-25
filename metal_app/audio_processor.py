# metal_app/audio_processor.py
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'metal_app', 'FINALtry_metal_prediction_MyData.h5')
SR = 16000
FFT_SIZE = 4096

model = load_model(MODEL_PATH)
CLASS_NAMES = ["Stainless-Steel","Brass"]

def predict_metal_from_file(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR, mono=True)
        
        # Find hit
        energy = librosa.feature.rms(y=y, hop_length=512)[0]
        spike = np.argmax(energy) * 512
        start = max(0, spike - 1024)
        clip = y[start:start + 2048]
        
        # FFT
        fft = np.abs(np.fft.rfft(clip, n=FFT_SIZE))
        fft = fft / (np.max(fft) + 1e-8)
        fft = np.log1p(fft)
        
        features = fft[np.newaxis, ..., np.newaxis]
        
        pred = model.predict(features, verbose=0)[0][0]
        result = CLASS_NAMES[0] if pred > 0.5 else CLASS_NAMES[1]
        conf = pred if result == CLASS_NAMES[0] else 1 - pred
        
        print(f"[PREDICT] {result} | Conf: {conf:.3f}")
        
        return {
            'metal': result,
            'confidence': round(float(conf), 3)
        }
    
    except Exception as e:
        return {'error': str(e)}