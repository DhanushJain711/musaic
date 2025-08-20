from music21 import converter, note

def extract_expected_notes(musicxml_path, tempo, measure_number):
    """Extract expected pitches and approximate timings from MusicXML"""
    score = converter.parse(musicxml_path)
    expected_notes = []
    
    current_time = 0.0
    quarter_note_duration = 60.0 / tempo  # seconds per quarter note
    
    part = score.parts[0]  # Assuming the first part is the one we want
    measures = part.getElementsByClass('Measure')
    selected_measures = measures[:measure_number] if measure_number <= len(measures) else measures
    for measure in selected_measures:
        for element in measure.notesAndRests:
            if isinstance(element, note.Note):
                pitch_midi = element.pitch.midi  # MIDI note number
                duration = element.duration.quarterLength * quarter_note_duration
                
                expected_notes.append({
                    'pitch_midi': pitch_midi,
                    'start_time': current_time,
                    'duration': duration,
                    'pitch_name': element.pitch.nameWithOctave
                })
                current_time += duration
            elif isinstance(element, note.Rest):
                # Skip rests but advance time
                duration = element.duration.quarterLength * quarter_note_duration
                current_time += duration

    return expected_notes

if __name__ == "__main__":
    print(extract_expected_notes('/Users/dhanush/downloads/trumpet_etude_music.musicxml', 44, 4))