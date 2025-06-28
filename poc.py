import madmom 

class SpectralFlux():
    def __init__(self, wav_file):
        spec = madmom.audio.spectrogram.Spectrogram(wav_file,frame_size=2048,hop_size=200,fft_size=4096)
        self.sf = madmom.features.onsets.spectral_flux(spec)
        
        