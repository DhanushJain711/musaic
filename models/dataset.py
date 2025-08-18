import sqlite3
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import dotenv
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import librosa
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoodSoundsDatabase:
    """Handler for the GOOD-SOUNDS SQLite database"""
    
    def __init__(self, dataset_path: str):
        """Initialize database connection"""
        self.dataset_path = f'{dataset_path}/sound_files'
        self.db_path = f'{dataset_path}/database.sqlite'
        self.connection = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def get_all_sounds(self) -> pd.DataFrame:
        """Get all sounds from the database"""
        query = "SELECT * FROM sounds"
        df = pd.read_sql_query(query, self.connection)
        #Remove recordings that are scales; we want individual notes for now
        df = df[~df['klass'].str.contains('scale', case=False, na=False)]
        df = self._add_file_info(df)
        return df
    
    def get_sounds_by_klass(self, klass: str) -> pd.DataFrame:
        """ Get sounds filtered by klass (note property label) """
        query = "SELECT * FROM sounds WHERE klass = ?"
        df = pd.read_sql_query(query, self.connection, params=[klass])
        df = self._add_file_info(df)
        return df
    
    def get_sounds_by_instrument(self, instrument: str) -> pd.DataFrame:
        """Get sounds filtered by instrument"""
        query = "SELECT * FROM sounds WHERE instrument = ?"
        df = pd.read_sql_query(query, self.connection, params=[instrument])
        #Remove recordings that are scales; we want individual notes for now
        df = df[~df['klass'].str.contains('scale', case=False, na=False)]
        df = self._add_file_info(df)
        return df

    def get_sounds_by_klass_and_instrument(self, klass: str, instrument: str) -> pd.DataFrame:
        """Get sounds filtered by both klass and instrument"""
        query = "SELECT * FROM sounds WHERE klass = ? AND instrument = ?"
        df = pd.read_sql_query(query, self.connection, params=[klass, instrument])
        #Remove recordings that are scales; we want individual notes for now
        df = df[~df['klass'].str.contains('scale', case=False, na=False)]
        df = self._add_file_info(df)
        return df
    
    def get_unique_klasses(self) -> List[str]:
        """Get all unique klass values"""
        query = "SELECT DISTINCT klass FROM sounds WHERE klass IS NOT NULL"
        df = pd.read_sql_query(query, self.connection)
        return df['klass'].tolist()
    
    def get_unique_instruments(self) -> List[str]:
        """Get all unique instrument values"""
        query = "SELECT DISTINCT instrument FROM sounds WHERE instrument IS NOT NULL"
        df = pd.read_sql_query(query, self.connection)
        return df['instrument'].tolist()
    
    def _add_file_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add file path info to the dataframe"""
        def get_pack_path(pack_id: int) -> str:
            """Get the pack name from the pack_id"""
            query = "SELECT name FROM packs WHERE id = ?"
            self.cursor.execute(query, (pack_id,))
            result = self.cursor.fetchone()
            
            if result is None:
                logger.warning(f"Pack ID {pack_id} not found in packs table")
                return None
            
            pack_name = result[0]
            pack_full_path = f'{self.dataset_path}/{pack_name}'
            
            # Check if pack directory exists
            if not os.path.exists(pack_full_path):
                logger.warning(f"Pack directory not found: {pack_full_path}")
                return None

            #Get subfolders (recording setup)
            try:
                subfolders = [name for name in os.listdir(pack_full_path) 
                            if os.path.isdir(os.path.join(pack_full_path, name))]
                if not subfolders:
                    logger.warning(f"No subfolders found in {pack_full_path}")
                    return pack_full_path  # Return the pack path directly
                
                return f'{pack_full_path}/{subfolders[0]}' #Just use first subfolder for now
            except OSError as e:
                logger.error(f"Error accessing directory {pack_full_path}: {e}")
                return None
        
        #Get paths to the packs
        df['file_path'] = df['pack_id'].apply(get_pack_path)
        
        # Filter out rows where pack_path couldn't be determined
        initial_count = len(df)
        df = df.dropna(subset=['file_path'])
        filtered_count = len(df)
        
        if filtered_count < initial_count:
            logger.warning(f"Filtered out {initial_count - filtered_count} records with missing pack information")
        
        #Append the wav file names
        df['file_path'] = df['file_path'] + '/' + df['pack_filename']
        
        # Filter out rows where the actual audio file doesn't exist
        def file_exists(file_path):
            return os.path.exists(file_path)
        
        df['file_exists'] = df['file_path'].apply(file_exists)
        missing_files = df[~df['file_exists']]
        if len(missing_files) > 0:
            logger.warning(f"Found {len(missing_files)} records with missing audio files")
            # Log a few examples
            for i, (_, row) in enumerate(missing_files.head(3).iterrows()):
                logger.warning(f"  Missing file: {row['file_path']}")
        
        df = df[df['file_exists']].drop('file_exists', axis=1)
        
        return df
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 


class GoodSoundsDataset(Dataset):
    """Custom Dataset for loading audio files and converting to spectrograms"""
    
    def __init__(self, dataframe, spectrogram_function, label_encoder=None, 
                 cache_spectrograms=False, cache_dir=None):
        """
        Args:
            dataframe: pandas DataFrame with 'file_path' and 'klass' columns
            spectrogram_function: function to convert audio to spectrogram
            label_encoder: sklearn LabelEncoder (fit on training data)
            cache_spectrograms: whether to cache spectrograms to disk
            cache_dir: directory to save cached spectrograms
        """
        self.df = dataframe.reset_index(drop=True)
        self.spectrogram_function = spectrogram_function
        self.label_encoder = label_encoder
        self.cache_spectrograms = cache_spectrograms
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_spectrograms and self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        label = row['klass']
        
        # Try to load cached spectrogram first
        cache_path = None
        if self.cache_spectrograms and self.cache_dir:
            cache_filename = f"spec_{hash(file_path)}_{idx}.pt"
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                try:
                    spectrogram = torch.load(cache_path)
                except:
                    # If cache is corrupted, regenerate
                    spectrogram = self._load_and_convert_audio(file_path)
                    torch.save(spectrogram, cache_path)
            else:
                spectrogram = self._load_and_convert_audio(file_path)
                torch.save(spectrogram, cache_path)
        else:
            spectrogram = self._load_and_convert_audio(file_path)
        
        # Encode label
        if self.label_encoder:
            label_encoded = self.label_encoder.transform([label])[0]
        else:
            label_encoded = label
        
        return {
            'spectrogram': spectrogram,
            'label': torch.tensor(label_encoded, dtype=torch.long),
            'file_path': file_path,
            'original_label': label
        }
    
    def _load_and_convert_audio(self, file_path):
        """Load audio file and convert to spectrogram"""
        try:
            # Convert to spectrogram
            spectrogram = self.spectrogram_function(file_path)
            
            # Ensure it's a tensor
            if not isinstance(spectrogram, torch.Tensor):
                spectrogram = torch.from_numpy(spectrogram).float()
            
            return spectrogram
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros as fallback
            return torch.zeros((128, 128))  # Adjust dimensions as needed
