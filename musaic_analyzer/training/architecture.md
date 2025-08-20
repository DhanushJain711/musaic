# Musical Note Classifier - Architecture & Implementation Guide

## Project Overview
A deep learning system for classifying individual musical notes from audio recordings across 12 different instruments, handling ~100 different quality labels (attack quality, pitch stability, timbre, etc.).

### Key Challenges
- **Variable note lengths**: 0.25 seconds (staccato) to 5+ seconds (sustained)
- **Multiple instruments**: 12 different instruments with distinct characteristics
- **Large label space**: ~100 hierarchical labels (good-sound, bad-attack-tongue, etc.)
- **Limited data**: ~8000 recordings total (~650-700 per instrument)
- **CPU-only training**: Must be efficient without GPU

## Dataset Structure
```
Total samples: 8000
Instruments: 12 (trumpet, flute, violin, clarinet, etc.)
Samples per instrument: ~650-700
Labels: ~100 classes (hierarchical structure)
  - Quality: good/regular/bad
  - Aspect: attack/pitch/timbre/stability/dynamics/richness/air
  - Specific: vibrato/tongue/pressure/ponticello/etc.
```

## Model Architecture

### Complete Architecture
```python
class NoteAnalyzer(nn.Module):
    def __init__(self, num_instruments=12, num_classes=100):
        super().__init__()
        
        # 1. Instrument Embedding Layer
        # Maps instrument ID to learned 64-dim representation
        self.instrument_embedding = nn.Embedding(num_instruments, 64)
        
        # 2. CNN Feature Extractor
        # Processes variable-length spectrograms
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, None))  # Pool frequency only, preserve time
        )
        
        # 3. Temporal Processing Layer
        # Aggregates features across time (Conv1d faster than GRU on CPU)
        self.temporal = nn.Conv1d(256*8, 512, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Fusion Layer
        # Combines audio features with instrument embedding
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 512),  # features + instrument
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 5. Classification Head
        self.classifier = nn.Linear(512, num_classes)
```

### Simplified Starter Version
```python
class SimpleStarterModel(nn.Module):
    """Simpler version to start with - trains faster"""
    def __init__(self, num_instruments=12, num_classes=100):
        super().__init__()
        
        self.instrument_embedding = nn.Embedding(num_instruments, 32)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

## Inference Process

### 1. Audio Preprocessing
```python
def preprocess_audio(audio, sr=22050):
    # Compute spectrogram
    n_fft = 512  # ~23ms window
    hop_length = 128  # ~6ms hop
    
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec))
    
    # Normalize
    spec_db = (spec_db - spec_db.mean()) / spec_db.std()
    
    return torch.FloatTensor(spec_db).unsqueeze(0)  # Add channel dim
```

### 2. Forward Pass
```python
def inference(model, audio, instrument_id, note_duration):
    model.eval()
    
    with torch.no_grad():
        # 1. Prepare input
        spectrogram = preprocess_audio(audio)
        spectrogram = spectrogram.unsqueeze(0)  # Add batch dim
        
        # 2. Extract CNN features
        features = model.cnn(spectrogram)
        
        # 3. Temporal processing (adaptive based on duration)
        if note_duration < 0.5:  # Short note
            # Focus on attack (first 50-100ms)
            features = features[:, :, :, :10]  # First 10 time frames
        
        features = features.reshape(1, -1, features.shape[-1])
        temporal_features = model.temporal(features)
        pooled = model.temporal_pool(temporal_features).squeeze()
        
        # 4. Add instrument context
        inst_emb = model.instrument_embedding(torch.tensor([instrument_id]))
        
        # 5. Fusion and classification
        combined = torch.cat([pooled, inst_emb.squeeze()], dim=0)
        fused = model.fusion(combined.unsqueeze(0))
        logits = model.classifier(fused)
        
        # 6. Get prediction
        pred_class = torch.argmax(logits, dim=1)
        confidence = torch.softmax(logits, dim=1).max()
        
    return pred_class.item(), confidence.item()
```

### 3. Batch Processing for Fast Passages
```python
def batch_inference(model, audio_segments, instrument_ids):
    """Process multiple notes efficiently"""
    
    # Prepare batch
    specs = [preprocess_audio(audio) for audio in audio_segments]
    
    # Pad to same length for batching
    max_len = max(s.shape[-1] for s in specs)
    padded = [F.pad(s, (0, max_len - s.shape[-1])) for s in specs]
    
    batch_specs = torch.stack(padded)
    batch_instruments = torch.tensor(instrument_ids)
    
    # Single forward pass
    with torch.no_grad():
        predictions = model(batch_specs, batch_instruments)
    
    return predictions
```

## Training Process

### 1. Data Preparation
```python
class NoteDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, labels, instruments, augment=True):
        self.audio_files = audio_files
        self.labels = labels
        self.instruments = instruments
        self.augment = augment
        
        # Map instrument names to IDs
        self.instrument_map = {
            'trumpet': 0, 'flute': 1, 'violin': 2, 'clarinet': 3,
            # ... etc for all 12 instruments
        }
        
        # Map label strings to class IDs
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    def __getitem__(self, idx):
        # Load audio
        audio, sr = librosa.load(self.audio_files[idx], sr=22050)
        
        # Augmentation (crucial for 80%+ accuracy)
        if self.augment:
            audio = self.augment_audio(audio, sr)
        
        # Compute spectrogram
        spec = preprocess_audio(audio)
        
        # Get labels
        instrument_id = self.instrument_map[self.instruments[idx]]
        label_id = self.label_map[self.labels[idx]]
        duration = len(audio) / sr
        
        return spec, instrument_id, duration, label_id
    
    def augment_audio(self, audio, sr):
        if np.random.random() > 0.5:
            # Pitch shift
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
        if np.random.random() > 0.5:
            # Time stretch
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        if np.random.random() > 0.5:
            # Add noise
            noise = np.random.normal(0, 0.002, len(audio))
            audio = audio + noise
        
        return audio
```

### 2. Custom Collate Function for Batching
```python
def collate_fn(batch):
    """Handle variable-length spectrograms in batches"""
    specs, instruments, durations, labels = zip(*batch)
    
    # Find max length in batch
    max_len = max(spec.shape[-1] for spec in specs)
    
    # Pad all to same length
    padded_specs = []
    for spec in specs:
        if spec.shape[-1] < max_len:
            padding = max_len - spec.shape[-1]
            padded = F.pad(spec, (0, padding), value=0)
        else:
            padded = spec
        padded_specs.append(padded)
    
    return (torch.stack(padded_specs),
            torch.tensor(instruments),
            torch.tensor(durations),
            torch.tensor(labels))
```

### 3. Training Loop
```python
def train_model(model, train_dataset, val_dataset, epochs=30):
    # CPU optimizations
    torch.set_num_threads(8)  # Use all CPU cores
    torch.set_flush_denormal(True)
    
    # Data loaders with batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Loss function with class weights for imbalanced data
    class_weights = calculate_class_weights(train_dataset.labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (specs, inst_ids, durations, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(specs, inst_ids, durations)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for specs, inst_ids, durations, labels in val_loader:
                outputs = model(specs, inst_ids, durations)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model
```

## Performance Expectations

### Training Time (8-core CPU, 32GB RAM)
- **Simple Model**: 5-7 hours for 30 epochs
- **Full Model**: 10-15 hours for 30 epochs
- **Per epoch**: 10-30 minutes depending on model complexity

### Expected Accuracy
- **Without augmentation**: 60-70%
- **With augmentation**: 70-80%
- **With class balancing**: +5-10%
- **Final expected**: 75-85%

### Common Issues by Instrument
- **Wind instruments**: Attack classification most important
- **String instruments**: Bow-specific labels (ponticello, sul tasto)
- **Brass**: Breath and tonguing issues

## Future Optimizations

### For Fast Passages (Phase 2)
- Implement sequence model for analyzing multiple notes at once
- Use sliding window approach for real-time analysis
- Consider lighter model for rapid inference

### For Production (Phase 3)
- ONNX export for 2-5x faster CPU inference
- Implement caching for repeated notes
- Add confidence thresholds for uncertain predictions

## Key Design Decisions

1. **Conv1d over GRU**: 5-10x faster on CPU with similar performance
2. **Instrument embeddings**: Critical for handling timbral differences
3. **Adaptive pooling**: Handles any note length without retraining
4. **Heavy augmentation**: Effectively multiplies dataset size
5. **Class weighting**: Essential for imbalanced labels (some classes have <20 samples)

## Implementation Checklist

- [ ] Prepare dataset with proper train/val/test splits
- [ ] Implement data augmentation pipeline
- [ ] Create custom collate function for batching
- [ ] Build SimpleStarterModel first
- [ ] Train for 30 epochs with monitoring
- [ ] Analyze per-class and per-instrument performance
- [ ] Implement full model if needed
- [ ] Export to ONNX for faster inference