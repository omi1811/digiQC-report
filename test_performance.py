#!/usr/bin/env python3
"""
Simple performance test to validate optimizations work correctly.
"""
import pandas as pd
import time
from io import StringIO

# Test data
test_csv = """Location L0,Project,Project Name,Stage,Date,Eqc Type
Itrend City Life,CityLife,City Life Project,Pre,01-01-2024,RCC Check
Itrend Futura,Futura,Futura Project,During,02-01-2024,Masonry
Itrend Palacio,Palacio,Palacio Project,Post,03-01-2024,Plaster
Test Project,TestProj,Test Project Name,Pre,04-01-2024,Tiling
"""

def test_canonical_project_vectorized():
    """Test that the vectorized canonical_project function works correctly."""
    print("Testing canonical_project_vectorized...")
    
    # Import after module is available
    import app
    
    # Create test dataframe
    df = pd.read_csv(StringIO(test_csv))
    
    # Test vectorized version
    start = time.time()
    result_vec = app.canonical_project_vectorized(df)
    time_vec = time.time() - start
    
    # Test original (row-by-row) version
    start = time.time()
    result_orig = df.apply(app.canonical_project, axis=1)
    time_orig = time.time() - start
    
    # Verify results match
    assert (result_vec == result_orig).all(), "Results don't match!"
    
    print(f"  ✓ Results match")
    print(f"  Original time: {time_orig*1000:.2f}ms")
    print(f"  Vectorized time: {time_vec*1000:.2f}ms")
    if time_orig > 0:
        print(f"  Speedup: {time_orig/time_vec:.2f}x")
    print()

def test_file_caching():
    """Test that file caching mechanism works."""
    print("Testing file caching...")
    
    import app
    import os
    
    # Clear cache
    app._file_cache.clear()
    
    # First call - should populate cache
    def check_py(f):
        return f.endswith('.py')
    
    start = time.time()
    result1 = app._find_files_cached(check_py, "test_py")
    time1 = time.time() - start
    
    # Second call - should use cache
    start = time.time()
    result2 = app._find_files_cached(check_py, "test_py")
    time2 = time.time() - start
    
    # Verify results match
    assert result1 == result2, "Cached results don't match!"
    
    print(f"  ✓ Cache working correctly")
    print(f"  First call: {time1*1000:.2f}ms")
    print(f"  Cached call: {time2*1000:.2f}ms")
    if time1 > 0:
        print(f"  Speedup: {time1/time2:.2f}x")
    print()

def test_coerce_numeric():
    """Test optimized coerce_numeric_df function."""
    print("Testing coerce_numeric_df optimization...")
    
    from combine_reports_to_excel import coerce_numeric_df
    
    # Create test dataframe
    test_data = {
        'Checklists': ['Item 1', 'Item 2', 'Item 3'],
        'Building': ['A', 'B', 'C'],
        'Pre': ['10', '20', '30'],
        'During': ['5', '15', '25'],
        'Post': ['1', '2', '3']
    }
    df = pd.DataFrame(test_data)
    
    start = time.time()
    result = coerce_numeric_df(df)
    elapsed = time.time() - start
    
    # Verify numeric columns are converted
    assert result['Pre'].dtype == 'int64', "Pre column not converted to int"
    assert result['During'].dtype == 'int64', "During column not converted to int"
    assert result['Post'].dtype == 'int64', "Post column not converted to int"
    
    # Verify text columns unchanged
    assert result['Checklists'].dtype == 'object', "Checklists should stay as object"
    assert result['Building'].dtype == 'object', "Building should stay as object"
    
    print(f"  ✓ Numeric conversion working correctly")
    print(f"  Processing time: {elapsed*1000:.2f}ms")
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("Performance Optimization Tests")
    print("=" * 60)
    print()
    
    try:
        test_canonical_project_vectorized()
        test_file_caching()
        test_coerce_numeric()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
