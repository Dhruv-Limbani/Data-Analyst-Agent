from mcp.server.fastmcp import FastMCP

import pandas as pd
import os
import chardet
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for metadata
METADATA_CACHE = {}

def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file using chardet library.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding as string
    """
    try:
        with open(file_path, 'rb') as file:
            # Read a sample of the file for encoding detection
            raw_data = file.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fall back to common encodings if confidence is low
            if confidence < 0.7:
                common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for enc in common_encodings:
                    try:
                        with open(file_path, 'r', encoding=enc) as test_file:
                            test_file.read(1000)  # Try to read a small portion
                        logger.info(f"Successfully validated encoding: {enc}")
                        return enc
                    except UnicodeDecodeError:
                        continue
            
            return encoding or 'utf-8'
            
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}. Using utf-8 as fallback.")
        return 'utf-8'

def safe_read_csv(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file with automatic encoding detection and error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional pandas.read_csv parameters
        
    Returns:
        DataFrame if successful, None otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    # Detect encoding
    encoding = detect_encoding(file_path)
    
    # Try reading with detected encoding
    encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for enc in encodings_to_try:
        try:
            logger.info(f"Attempting to read CSV with encoding: {enc}")
            df = pd.read_csv(
                file_path, 
                encoding=enc,
                low_memory=False,  # Avoid dtype warnings for mixed types
                na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan'],  # Common missing value representations
                **kwargs
            )
            logger.info(f"Successfully read CSV with {enc} encoding")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read with {enc} encoding: {e}")
            continue
        except Exception as e:
            logger.error(f"Error reading CSV with {enc} encoding: {e}")
            continue
    
    logger.error("Failed to read CSV with all attempted encodings")
    return None

def get_column_metadata(col_data: pd.Series) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a single column.
    
    Args:
        col_data: Pandas Series representing a column
        
    Returns:
        Dictionary containing column metadata
    """
    col_meta = {
        "type": str(col_data.dtype),
        "missing_values": int(col_data.isnull().sum()),
        "missing_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
        "non_null_count": int(col_data.count()),
        "unique_count": int(col_data.nunique())
    }
    
    # Add numeric statistics for numeric columns
    if pd.api.types.is_numeric_dtype(col_data):
        try:
            stats = col_data.describe()
            col_meta["stats"] = {
                "mean": float(stats['mean']) if pd.notna(stats['mean']) else None,
                "std": float(stats['std']) if pd.notna(stats['std']) else None,
                "min": float(stats['min']) if pd.notna(stats['min']) else None,
                "25%": float(stats['25%']) if pd.notna(stats['25%']) else None,
                "50%": float(stats['50%']) if pd.notna(stats['50%']) else None,
                "75%": float(stats['75%']) if pd.notna(stats['75%']) else None,
                "max": float(stats['max']) if pd.notna(stats['max']) else None
            }
        except Exception as e:
            logger.warning(f"Failed to compute numeric stats: {e}")
            col_meta["stats"] = None
    else:
        # For non-numeric columns, add mode and frequency info
        col_meta["stats"] = None
        if col_data.count() > 0:  # Only if there are non-null values
            try:
                mode_info = col_data.mode()
                if not mode_info.empty:
                    col_meta["most_frequent"] = str(mode_info.iloc[0])
                    col_meta["most_frequent_count"] = int(col_data.value_counts().iloc[0])
            except Exception as e:
                logger.warning(f"Failed to compute mode: {e}")
    
    # Sample values (safely handle different data types)
    try:
        non_null_values = col_data.dropna()
        if len(non_null_values) > 0:
            unique_values = non_null_values.unique()[:5]
            col_meta["sample_values"] = [str(val) for val in unique_values]
        else:
            col_meta["sample_values"] = []
    except Exception as e:
        logger.warning(f"Failed to extract sample values: {e}")
        col_meta["sample_values"] = []
    
    return col_meta

# Constants
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
UPLOAD_DIRECTORY = os.path.join(SCRIPT_DIR, "temp_data_files")
PLOT_DIRECTORY = os.path.join(SCRIPT_DIR, "temp_plot_images")

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

# Initialize FastMCP
mcp = FastMCP("CSV Data Analyzer")

def resolve_file_path(filename: str) -> str:
    """
    Resolve file path, checking both temp_data_files directory and absolute paths.
    
    Args:
        filename: Filename or path to resolve
        
    Returns:
        Absolute path to the file
    """
    # If it's just a filename, look in temp_data_files
    if os.path.basename(filename) == filename:
        return os.path.abspath(os.path.join(UPLOAD_DIRECTORY, filename))
    
    # If it's already a path, resolve it
    abs_path = os.path.abspath(filename)
    
    # If the path doesn't exist, try looking in temp_data_files
    if not os.path.exists(abs_path):
        temp_path = os.path.abspath(os.path.join(UPLOAD_DIRECTORY, os.path.basename(filename)))
        if os.path.exists(temp_path):
            return temp_path
    
    return abs_path

@mcp.tool()
def analyze_csv_metadata(
    filename: str,
    use_cache: bool = True,
    nrows: Optional[int] = None,
    columns_subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a CSV file uploaded by the user and generate comprehensive metadata for each column.

    Purpose:
        This tool is designed to inspect CSV files stored in the temp_data_files directory.
        It provides detailed insights into each column, including data type, missing values,
        unique values, sample entries, and statistical summaries for numeric columns.
        This metadata helps the main LLM or downstream tools decide how to further process
        or analyze the dataset.

    Args:
        filename (str):
            The name of the CSV file to analyze. Can be a simple filename (e.g., "data.csv")
            or a relative path. The tool prioritizes the temp_data_files directory where user
            uploads are stored.
        use_cache (bool, default=True):
            If True, previously computed metadata for the same file and parameters will be
            reused from the cache, improving performance. Set to False to force fresh analysis.
        nrows (Optional[int], default=None):
            Number of rows to read from the CSV file. If None, the entire file is read.
            Useful for previewing or sampling large datasets.
        columns_subset (Optional[List[str]], default=None):
            A list of specific column names to analyze. If None, all columns are analyzed.

    Returns:
        Dict[str, Any]:
            A dictionary containing the analysis results, structured as follows:
            - success (bool): Whether the analysis was successful.
            - dataset_info (dict):
                * file_path: Absolute path to the CSV file.
                * file_size_mb: File size in megabytes.
                * rows: Number of rows in the dataset.
                * columns: Number of columns in the dataset.
                * memory_usage_mb: Memory usage of the DataFrame in MB.
                * column_names: List of all column names.
            - column_metadata (dict): Per-column analysis including:
                * data type
                * number of missing values
                * number of unique values
                * sample values
                * summary statistics (for numeric columns: mean, min, max, std, quartiles)
                * any errors encountered during analysis
            - analysis_parameters (dict):
                * rows_analyzed: Number of rows actually analyzed.
                * columns_analyzed: List of columns analyzed.
                * used_cache: Whether cached results were used.

    Notes:
        - If the file does not exist or cannot be read, `success` will be False and an
          `error` message will be included.
        - Metadata is cached by default to improve performance for repeated analyses.
        - Designed to integrate seamlessly with ReAct-style agents and MCP servers,
          allowing LLMs to call this tool and reason about CSV contents without
          manually reading the file.
    """
    try:
        # Resolve file path (prioritize temp_data_files directory)
        abs_path = resolve_file_path(filename)
        cache_key = f"{abs_path}_{nrows}_{columns_subset}"
        
        # Check cache if enabled
        if use_cache and cache_key in METADATA_CACHE:
            logger.info("Returning cached metadata")
            return METADATA_CACHE[cache_key]
        
        # Validate file existence
        if not os.path.exists(abs_path):
            return {
                "success": False,
                "error": f"File not found: {abs_path}. Make sure the file is uploaded to temp_data_files directory.",
                "file_path": abs_path,
                "searched_locations": [
                    os.path.join(UPLOAD_DIRECTORY, filename),
                    abs_path
                ]
            }
        
        # Read CSV with optional parameters
        read_kwargs = {}
        if nrows is not None:
            read_kwargs['nrows'] = nrows
        if columns_subset is not None:
            read_kwargs['usecols'] = columns_subset
            
        df = safe_read_csv(abs_path, **read_kwargs)
        
        if df is None:
            return {
                "success": False,
                "error": f"Failed to read CSV file. Check file format and encoding.",
                "file_path": abs_path
            }
        
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Generate metadata for each column
        column_metadata = {}
        for col in df.columns:
            try:
                column_metadata[col] = get_column_metadata(df[col])
            except Exception as e:
                logger.error(f"Failed to process column '{col}': {e}")
                column_metadata[col] = {"error": f"Failed to process column: {str(e)}"}
        
        # Compile comprehensive results
        result = {
            "success": True,
            "dataset_info": {
                "file_path": abs_path,
                "file_size_mb": round(os.path.getsize(abs_path) / (1024 * 1024), 2),
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "column_names": list(df.columns)
            },
            "column_metadata": column_metadata,
            "analysis_parameters": {
                "rows_analyzed": nrows or df.shape[0],
                "columns_analyzed": columns_subset or list(df.columns),
                "used_cache": False
            }
        }
        
        # Cache results if enabled
        if use_cache:
            METADATA_CACHE[cache_key] = result
            
        logger.info("Metadata analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error during CSV analysis: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "file_path": filename
        }

@mcp.tool()
def list_uploaded_csv_files() -> Dict[str, Any]:
    """
    List all CSV files that have been uploaded by the user in the temp_data_files directory.

    Purpose:
        This tool helps the main LLM or ReAct agent discover which CSV files are available
        for analysis. It provides metadata about each file, allowing the LLM to reason about
        file selection, prioritization, and further processing. Useful when the user has not
        specified a file, or when multiple CSVs are present.

    Args:
        None

    Returns:
        Dict[str, Any]:
            A dictionary containing information about uploaded CSV files:
            - success (bool): Whether the operation was successful.
            - directory (str): Absolute path to the upload directory.
            - csv_files (list[dict]): List of CSV files with the following metadata for each:
                * filename: Name of the CSV file.
                * full_path: Absolute file path.
                * size_mb: File size in megabytes.
                * size_kb: File size in kilobytes.
                * modified_timestamp: Last modification timestamp (epoch seconds).
            - count (int): Number of CSV files found.
            - message (str): Informative message about the result.
            - error (str, optional): Error message if operation failed.

    Notes:
        - If the upload directory does not exist, it will be automatically created.
        - If the path exists but is not a directory, the tool will return an error.
        - The list of CSV files is sorted by modification time (newest first).
        - This tool does not read file contents; it only provides metadata.
        - Designed for integration with LLM agents to guide file selection
          and downstream data analysis workflows.
    """
    try:
        upload_dir = os.path.abspath(UPLOAD_DIRECTORY)
        
        # Create directory if it doesn't exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
            return {
                "success": True,
                "directory": upload_dir,
                "csv_files": [],
                "count": 0,
                "message": f"Created upload directory: {upload_dir}"
            }
        
        if not os.path.isdir(upload_dir):
            return {
                "success": False,
                "error": f"Upload path exists but is not a directory: {upload_dir}",
                "directory": upload_dir
            }
        
        csv_files = []
        
        for csv_path in Path(upload_dir).glob("*.csv"):
            if csv_path.is_file():
                try:
                    file_stat = csv_path.stat()
                    csv_files.append({
                        "filename": csv_path.name,
                        "full_path": str(csv_path),
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "size_kb": round(file_stat.st_size / 1024, 2),
                        "modified_timestamp": file_stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Failed to get stats for {csv_path}: {e}")
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: x["modified_timestamp"], reverse=True)
        
        return {
            "success": True,
            "directory": upload_dir,
            "csv_files": csv_files,
            "count": len(csv_files),
            "message": f"Found {len(csv_files)} CSV files in upload directory"
        }
        
    except Exception as e:
        logger.error(f"Error listing uploaded CSV files: {e}")
        return {
            "success": False,
            "error": f"Error listing uploaded CSV files: {str(e)}",
            "directory": UPLOAD_DIRECTORY
        }

@mcp.tool()
def list_csv_files(directory_path: str = ".", recursive: bool = False) -> Dict[str, Any]:
    """
    List all CSV files in a specified directory, optionally including subdirectories.

    Purpose:
        This tool allows the LLM or main agent to discover CSV files in arbitrary
        directories on the system. Useful as a fallback or supplementary tool when
        user uploads are not available or additional data sources are needed.

    Args:
        directory_path (str): Directory to search for CSV files. Default is the current
                              working directory.
        recursive (bool): If True, search all subdirectories recursively. Default is False.

    Returns:
        Dict[str, Any]:
            A dictionary containing information about CSV files found:
            - success (bool): Whether the operation was successful.
            - directory (str): Absolute path of the searched directory.
            - csv_files (list[dict]): List of CSV files with metadata:
                * filename: Name of the CSV file.
                * full_path: Absolute path to the file.
                * directory: Directory containing the file.
                * size_mb: File size in megabytes.
                * modified_timestamp: Last modification timestamp (epoch seconds).
            - count (int): Number of CSV files found.
            - recursive_search (bool): Whether the search was recursive.
            - error (str, optional): Error message if operation failed.

    Notes:
        - If the directory does not exist or is not a valid directory, the tool returns
          an error.
        - Files are sorted by size, largest first, to help prioritize analysis of
          potentially more important datasets.
        - This tool only lists metadata; it does not read file contents.
        - Designed for LLM integration to guide intelligent file selection and
          downstream CSV analysis workflows.
    """
    try:
        abs_dir = os.path.abspath(directory_path)
        
        if not os.path.exists(abs_dir):
            return {
                "success": False,
                "error": f"Directory not found: {abs_dir}",
                "directory": abs_dir
            }
        
        if not os.path.isdir(abs_dir):
            return {
                "success": False,
                "error": f"Path is not a directory: {abs_dir}",
                "directory": abs_dir
            }
        
        csv_files = []
        search_pattern = "**/*.csv" if recursive else "*.csv"
        
        for csv_path in Path(abs_dir).glob(search_pattern):
            if csv_path.is_file():
                try:
                    file_stat = csv_path.stat()
                    csv_files.append({
                        "filename": csv_path.name,
                        "full_path": str(csv_path),
                        "directory": str(csv_path.parent),
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "modified_timestamp": file_stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Failed to get stats for {csv_path}: {e}")
        
        # Sort by size (largest first)
        csv_files.sort(key=lambda x: x["size_mb"], reverse=True)
        
        return {
            "success": True,
            "directory": abs_dir,
            "csv_files": csv_files,
            "count": len(csv_files),
            "recursive_search": recursive
        }
        
    except Exception as e:
        logger.error(f"Error listing CSV files: {e}")
        return {
            "success": False,
            "error": f"Error listing CSV files: {str(e)}",
            "directory": directory_path
        }

@mcp.tool()
def get_csv_preview(
    filename: str, 
    nrows: int = 5, 
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a quick preview of a CSV file from user uploads.

    Purpose:
        This tool allows the LLM or main agent to quickly inspect the contents of a
        CSV file without performing full analysis. Useful for understanding the
        structure, column types, and sample data to plan further processing.

    Args:
        filename (str): Name of the CSV file to preview (located in temp_data_files
                        directory).
        nrows (int): Number of rows to include in the preview. Default is 5.
        columns (Optional[List[str]]): List of column names to include in the preview.
                                       Default is None (all columns are included).

    Returns:
        Dict[str, Any]:
            A dictionary containing preview information:
            - success (bool): Whether the operation succeeded.
            - filename (str): CSV filename.
            - file_path (str): Absolute path of the file.
            - preview_rows (int): Number of rows returned in preview.
            - total_columns (int): Total number of columns in the CSV.
            - column_names (List[str]): List of column names included in the preview.
            - data_types (Dict[str, str]): Data types of each column.
            - preview_data (List[Dict]): List of row dictionaries with column values.
            - error (str, optional): Error message if operation failed.
            - searched_locations (List[str], optional): Paths searched if file not found.

    Notes:
        - Converts all values to strings for uniform preview display; missing values
          are represented as None.
        - Does not modify the CSV file or cache results.
        - Designed for LLM integration to allow planning further analysis based on
          column names, types, and sample data.
        - Can be combined with `analyze_csv_metadata` to obtain full column statistics
          after inspecting the preview.
    """
    try:
        abs_path = resolve_file_path(filename)
        
        if not os.path.exists(abs_path):
            return {
                "success": False,
                "error": f"File not found: {filename}",
                "searched_locations": [
                    os.path.join(UPLOAD_DIRECTORY, filename),
                    abs_path
                ]
            }
        
        # Read limited data for preview
        read_kwargs = {"nrows": nrows}
        if columns:
            read_kwargs["usecols"] = columns
            
        df = safe_read_csv(abs_path, **read_kwargs)
        
        if df is None:
            return {
                "success": False,
                "error": "Failed to read CSV file for preview",
                "filename": filename
            }
        
        # Convert DataFrame to preview format
        preview_data = []
        for idx, row in df.iterrows():
            row_dict = {"row_index": int(idx)}
            for col in df.columns:
                row_dict[col] = str(row[col]) if pd.notna(row[col]) else None
            preview_data.append(row_dict)
        
        return {
            "success": True,
            "filename": filename,
            "file_path": abs_path,
            "preview_rows": len(preview_data),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview_data": preview_data
        }
        
    except Exception as e:
        logger.error(f"Error creating CSV preview: {e}")
        return {
            "success": False,
            "error": f"Error creating preview: {str(e)}",
            "filename": filename
        }

@mcp.tool()
def clear_metadata_cache() -> Dict[str, Any]:
    """
    Clear all cached CSV metadata to ensure fresh analysis on the next request.

    Purpose:
        This tool is useful when cached results may be outdated or if new CSV files 
        have been uploaded. Clearing the cache ensures that subsequent calls to 
        `analyze_csv_metadata` return up-to-date metadata.

    Args:
        None

    Returns:
        Dict[str, Any]:
            - success (bool): Indicates whether the cache was successfully cleared.
            - message (str): Descriptive message indicating how many cache entries were cleared.
            - cache_size_before (int): Number of cached entries before clearing.
            - cache_size_after (int): Number of cached entries after clearing (always 0).

    Notes:
        - This tool does not perform any analysis itself; it only clears internal cache.
        - Useful in interactive workflows when new files are uploaded or previous analysis 
          may be stale.
        - Can be called programmatically by the main LLM agent to ensure fresh metadata 
          before running complex analysis.
    """
    global METADATA_CACHE
    cache_size = len(METADATA_CACHE)
    METADATA_CACHE.clear()
    
    return {
        "success": True,
        "message": f"Cleared {cache_size} cached metadata entries",
        "cache_size_before": cache_size,
        "cache_size_after": 0
    }

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()