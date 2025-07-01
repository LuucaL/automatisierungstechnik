"""
Data transformation module for processing sensor data.
Includes basic data transformation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class DataTransformer:
    """Class for basic data transformation."""
    
    def __init__(self):
        pass
    
    def calculate_moving_average(self, df: pd.DataFrame, column: str, window: int = 5) -> pd.Series:
        """Calculate moving average for a time series column."""
        return df[column].rolling(window=window, min_periods=1).mean()
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert values between different units."""
        # Example conversion: Celsius to Fahrenheit
        if from_unit == 'C' and to_unit == 'F':
            return value * 9/5 + 32
        # Example conversion: grams to kilograms
        elif from_unit == 'grams' and to_unit == 'kg':
            return value / 1000
        else:
            return value  # No conversion if units not supported
    
    def aggregate_by_time(self, df: pd.DataFrame, time_freq: str = '1min') -> pd.DataFrame:
        """Aggregate data by time frequency (e.g., minute, hour)."""
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by time frequency and sensor type
        return df.groupby([pd.Grouper(key='timestamp', freq=time_freq), 'sensor_type']).agg({
            'value': ['mean', 'min', 'max', 'std']
        }).reset_index()

def main():
    """Demonstration of data transformation functionality."""
    print("Data transformation demonstration complete!")

if __name__ == "__main__":
    main()
