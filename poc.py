import librosa
import numpy as np
from madmom.features.onsets import OnsetPeakPickingProcessor, spectral_flux
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.signal import Signal


class SpectralFlux:
    def __init__(self, wav_file):
        self.wav_file = wav_file
        self.sample_rate = None
        self.onset_times = None
        self.audio_segments = []
        self.note_spectrograms = []

        self._process_file()

    def _process_file(self):
        waveforms, sampling_rate = librosa.load(self.wav_file, sr=None)
        self.sample_rate = sampling_rate

        spec = Spectrogram(self.wav_file)  # create spectrogram of the entire input
        sf = spectral_flux(spec)

        peak_picker = OnsetPeakPickingProcessor()  # detects onsets
        self.onset_times = peak_picker(sf)

        onset_samples = (self.onset_times * sampling_rate).astype(int)
        onset_samples = np.append(onset_samples, len(waveforms))

        for i in range(len(onset_samples) - 1):
            start = onset_samples[i]
            end = onset_samples[i + 1]
            segment = waveforms[start:end]
            self.audio_segments.append(segment)

            sig = Signal(segment, sample_rate=sampling_rate)
            note_spec = Spectrogram(sig)
            self.note_spectrograms.append(note_spec)  # spectrogram for each onset

    def get_onset_times(self):
        return self.onset_times

    def get_audio_segments(self):
        return self.audio_segments

    def get_note_spectrograms(self):
        return self.note_spectrograms


if __name__ == "__main__":
    extractor = SpectralFlux("data_sample.wav")
    print(f"onset times {extractor.get_onset_times()}")
    print(
        f"found {len(extractor.get_audio_segments())} audio segments/spectrgrams of each note"
    )

# TODO
# would be funny to make the for loop a list comp or somn but i seriously doubt it's worth/possible lol