"""
Test script to verify timestamp normalization works correctly.
"""
import pandas as pd
import numpy as np
from model.kronos import _normalize_timestamps, calc_time_stamps

print("Testing timestamp normalization...")

# Test 1: DatetimeIndex
print("\n1. Testing DatetimeIndex input...")
dt_index = pd.date_range('2024-01-01', periods=5, freq='H')
result = _normalize_timestamps(dt_index, 'test')
assert isinstance(result, pd.Series), "Should return a Series"
assert pd.api.types.is_datetime64_any_dtype(result), "Should be datetime dtype"
assert len(result) == 5, f"Expected length 5, got {len(result)}"
print("   ✓ Passed")

# Test 2: Series
print("\n2. Testing datetime Series input...")
dt_series = pd.Series(pd.date_range('2024-02-01', periods=3, freq='D'))
result = _normalize_timestamps(dt_series, 'test')
assert isinstance(result, pd.Series), "Should return a Series"
assert pd.api.types.is_datetime64_any_dtype(result), "Should be datetime dtype"
assert len(result) == 3, f"Expected length 3, got {len(result)}"
print("   ✓ Passed")

# Test 3: List of strings
print("\n3. Testing list of strings input...")
str_list = ['2024-03-01', '2024-03-02', '2024-03-03']
result = _normalize_timestamps(str_list, 'test')
assert isinstance(result, pd.Series), "Should return a Series"
assert pd.api.types.is_datetime64_any_dtype(result), "Should be datetime dtype"
assert len(result) == 3, f"Expected length 3, got {len(result)}"
print("   ✓ Passed")

# Test 4: calc_time_stamps with various inputs
print("\n4. Testing calc_time_stamps...")
for input_data in [
    pd.date_range('2024-01-01', periods=5, freq='H'),
    pd.Series(pd.date_range('2024-01-01', periods=5, freq='H')),
]:
    time_df = calc_time_stamps(input_data)
    assert 'minute' in time_df.columns
    assert 'hour' in time_df.columns
    assert 'weekday' in time_df.columns
    assert 'day' in time_df.columns
    assert 'month' in time_df.columns
    print(f"   ✓ {type(input_data).__name__} input works")

print("\n✅ All timestamp normalization tests passed!")
