# metal_app/forms.py
from django import forms

class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(widget=forms.FileInput(attrs={'accept': 'audio/wav'}))