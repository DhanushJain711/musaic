import sqlite3
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoodSoundsDatabase:
    """Handler for the GOOD-SOUNDS SQLite database"""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection
        
        Args:
            db_path: Path to the SQLite database file
        """
        dotenv.load_dotenv("connections.env")
        self.dataset_path = os.getenv('GOODSOUNDS_DATASET_PATH')
        self.db_path = db_path
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
    
    def get_sounds_by_klass(self, klass: str) -> pd.DataFrame:
        """ Get sounds filtered by klass (note property label) """
        query = "SELECT * FROM sounds WHERE klass = ?"
        return pd.read_sql_query(query, self.connection, params=[klass])
    
    def get_sounds_by_instrument(self, instrument: str) -> pd.DataFrame:
        """Get sounds filtered by instrument"""
        query = "SELECT * FROM sounds WHERE instrument = ?"
        df = pd.read_sql_query(query, self.connection, params=[instrument])
        #Remove recordings that are scales; we want individual notes for now
        df = df[~df['klass'].str.contains('scale', case=False, na=False)]
        df = self._add_file_info(df)
        return df
    
    def get_unique_klasses(self) -> List[str]:
        """Get all unique klass values"""
        query = "SELECT DISTINCT klass FROM sounds WHERE klass IS NOT NULL"
        df = pd.read_sql_query(query, self.connection)
        df = self._add_file_info(df)
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
            pack_name = self.cursor.fetchone()[0]
            return f'{self.dataset_path}/{pack_name}'
        
        #Get paths to the packs
        df['file_path'] = df['pack_id'].apply(get_pack_path)
        #Append the wav file names
        df['file_path'] = df['file_path'] + '/' + df['pack_filename']
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

db_service = GoodSoundsDatabase('/Users/dhanush/documents/musaic/good-sounds/database.sqlite')
df = db_service.get_sounds_by_instrument('trumpet')
print(df.head())
paths = df['file_path'].tolist()
print(paths[:10])