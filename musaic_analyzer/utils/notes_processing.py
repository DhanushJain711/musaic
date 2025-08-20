import librosa
import numpy as np
import madmom
from madmom.features.onsets import OnsetPeakPickingProcessor, spectral_flux
from madmom.audio.spectrogram import Spectrogram
from madmom.audio.signal import Signal
import logging
import os
import soundfile as sf

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
    
    def standard_segmentation(self, waveforms):
        sampling_rate = self.sample_rate

        # Slightly more sensitive onset detection
        onset_frames = librosa.onset.onset_detect(
            y=waveforms, 
            sr=sampling_rate,
            units='time',
            hop_length=512,
            backtrack=True,
            pre_max=0.02,      # Slightly more sensitive
            post_max=0.02,     
            pre_avg=0.07,      # Reduced further
            post_avg=0.07,     
            delta=0.04,        # More sensitive to catch missed boundaries
            wait=0.02          # Allow closer detection
        )

        # Fine-tune minimum duration - just slightly shorter
        min_duration = 0.06  # 60ms - should catch most real note boundaries
                                # but ignore vibrato fluctuations
        filtered_onsets = []

        for i, onset in enumerate(onset_frames):
            if i == 0:
                filtered_onsets.append(onset)
            else:
                if onset - filtered_onsets[-1] >= min_duration:
                    filtered_onsets.append(onset)

        onset_times = np.array(filtered_onsets)
        return onset_times

    def segment_with_pitches(self, waveforms):
        # Your existing onset detection (keep it as a starting point)
        onset_frames = librosa.onset.onset_detect(
            y=waveforms, 
            sr=self.sample_rate,
            units='time',
            hop_length=512,
            backtrack=True,
            pre_max=0.02,      # Reduced from 0.03
            post_max=0.02,     # Reduced from 0.03  
            pre_avg=0.08,      # Reduced from 0.10
            post_avg=0.08,     # Reduced from 0.10
            delta=0.05,        # Reduced from 0.07 (more sensitive)
            wait=0.02          # Reduced from 0.03
        )
        
        # Add pitch-based segmentation to catch missed boundaries
        pitches, magnitudes = librosa.piptrack(y=waveforms, sr=self.sample_rate)
        
        # Get fundamental frequency over time
        f0 = []
        times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=self.sample_rate)
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            f0.append(pitch if pitch > 0 else 0)
        
        # Find significant pitch changes
        f0 = np.array(f0)
        pitch_changes = []
        
        for i in range(1, len(f0)):
            if f0[i] > 0 and f0[i-1] > 0:  # Both frames have pitch
                # Detect significant pitch change (more than a semitone)
                cents_change = abs(1200 * np.log2(f0[i] / f0[i-1]))
                if cents_change > 100:  # ~1 semitone
                    pitch_changes.append(times[i])
        
        # Combine onset and pitch change detections
        all_boundaries = np.concatenate([onset_frames, pitch_changes])
        all_boundaries = np.unique(all_boundaries)
        all_boundaries = np.sort(all_boundaries)
        
        # Apply minimum duration filter
        min_duration = 0.10
        filtered_boundaries = [all_boundaries[0]] if len(all_boundaries) > 0 else []
        
        for boundary in all_boundaries[1:]:
            if boundary - filtered_boundaries[-1] >= min_duration:
                filtered_boundaries.append(boundary)
        
        onset_times = np.array(filtered_boundaries)
        return onset_times

    def segment_with_amplitudes(self, waveforms):
        # First pass: sensitive onset detection
        onset_frames_sensitive = librosa.onset.onset_detect(
            y=waveforms, sr=self.sample_rate, units='time',
            delta=0.03,        # More sensitive
            wait=0.01          # Allow closer onsets
        )

        # Second pass: filter intelligently
        min_very_short = 0.05   # 50ms - anything shorter is definitely noise
        min_normal = 0.12       # 120ms - normal minimum note duration
        
        filtered_onsets = []
        
        for i, onset in enumerate(onset_frames_sensitive):
            if i == 0:
                filtered_onsets.append(onset)
            else:
                time_diff = onset - filtered_onsets[-1]
                
                # Always keep if it's been long enough for a normal note
                if time_diff >= min_normal:
                    filtered_onsets.append(onset)
                # Keep if it's medium length AND there's a significant amplitude change
                elif time_diff >= min_very_short:
                    # Check for amplitude change at this boundary
                    onset_sample = int(onset * self.sample_rate)
                    window = int(0.02 * self.sample_rate)  # 20ms window
                    
                    pre_amp = np.mean(np.abs(waveforms[max(0, onset_sample-window):onset_sample]))
                    post_amp = np.mean(np.abs(waveforms[onset_sample:min(len(waveforms), onset_sample+window)]))
                    
                    # Keep if there's a significant change in amplitude
                    if abs(post_amp - pre_amp) > 0.01:  # Adjust this threshold
                        filtered_onsets.append(onset)
        
        onset_times = np.array(filtered_onsets)
        return onset_times

    def _process_file(self):
        waveforms, sampling_rate = librosa.load(self.wav_file, sr=None, mono=True)
        self.sample_rate = sampling_rate
        self.onset_times = self.segment_with_amplitudes(waveforms)
        
        print(f"Detected {len(self.onset_times)} onsets after filtering")
        onset_samples = (self.onset_times * self.sample_rate).astype(int)
        onset_samples = np.append(onset_samples, len(waveforms))

        for i in range(len(onset_samples) - 1):
            start = onset_samples[i]
            end = onset_samples[i + 1]
            segment = waveforms[start:end]
            self.audio_segments.append(segment)

            sig = Signal(segment, sample_rate=sampling_rate)
            note_spec = Spectrogram(sig)
            self.note_spectrograms.append(note_spec)  # spectrogram for each onset

        print(f"Created {len(self.note_spectrograms)} note spectrograms")

    def get_onset_times(self):
        return self.onset_times

    def get_audio_segments(self):
        return self.audio_segments
    
    def save_note_segments(self):
        directory = "musaic_analyzer/utils/note_segments"
        os.makedirs(directory, exist_ok=True)
        for i, note in enumerate(self.audio_segments):
            filename = f"{directory}/note_{i}.wav"
            sf.write(filename, note, self.sample_rate)

        print(f"Saved {len(self.audio_segments)} note segments to '{directory}' directory.")



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
    extractor = SpectralFlux("musaic_analyzer/utils/test_audios/trumpet_etude_clip1.wav")
    print(
        f"found {len(extractor.get_audio_segments())} spectrgrams of each note"
    )

    extractor.save_note_segments()

# TODO
# would be funny to make the for loop a list comp or somn but i seriously doubt it's worth/possible lol

# TODO
# ADD FUNCTIONALITY TO ACTUALLY SEPERATE EACH MUSICAL NOT(E)


