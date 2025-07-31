import librosa
import numpy as np
import madmom
from madmom.features.onsets import OnsetPeakPickingProcessor, spectral_flux
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.signal import Signal
import logging

logging.getLogger("madmom").setLevel(logging.ERROR)


def generate_spectrogram(wav_file):
    """Generate a spectrogram of a wav file"""
    waveforms, sampling_rate = librosa.load(wav_file, sr=None)
    spec = Spectrogram(wav_file)  # create spectrogram of the entire input
    return spec


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


class SuperFlux:  # spectral flux but with log filtering instead of linear filtering
    def __init__(self, num_bands=24, diff_max_bins=3, positive_diffs=True):
        """
        This init allows us to set optional parameters before doing log filtering
        """
        self.num_bands = num_bands
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(self, data):
        """
        runs actual superflux, which is log filtered instead of linearly filtered
        """
        spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(
            data, num_bands=self.num_bands
        )
        diff = madmom.audio.spectrogram.SpectrogramDifference(
            spec, diff_max_bins=self.diff_max_bins, positive_diffs=self.positive_diffs
        )
        return np.mean(diff, axis=1)


if __name__ == "__main__":
    extractor = SpectralFlux("data_sample.wav")
    print(f"onset times {extractor.get_onset_times()}")
    print(
        f"found {len(extractor.get_audio_segments())} spectrgrams of each note"
    )

    

# TODO
# would be funny to make the for loop a list comp or somn but i seriously doubt it's worth/possible lol

# TODO
# ADD FUNCTIONALITY TO ACTUALLY SEPERATE EACH MUSICAL NOT(E)


