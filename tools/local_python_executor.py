import io
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import ast
import os
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
import logging
import chardet
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
UPLOAD_DIRECTORY = os.path.join(SCRIPT_DIR, "temp_data_files")
PLOT_DIRECTORY = os.path.join(SCRIPT_DIR, "temp_plot_images")

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

ALLOWED_MODULES = {"pandas", "numpy", "matplotlib", "seaborn"}

mcp = FastMCP("Local Python Executor")

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

@mcp.tool()
def run_pandas_code(filename: str, code: str, nrows: int = 1000) -> Dict[str, Any]:
    """
    Safely execute user-provided pandas code on a CSV file in the upload directory.

    Purpose:
        This tool allows dynamic analysis and computation on CSV datasets using pandas.
        It provides a controlled and safe execution environment to prevent unsafe 
        operations while allowing the main LLM agent to perform custom data transformations,
        aggregations, or calculations.

    Args:
        filename (str): Name of the CSV file located in the temp_data_files directory.
        code (str): Python code (pandas-focused) to execute. 
                    - Must operate on the preloaded DataFrame named `df`.
                    - Restricted to safe operations; imports and unsafe functions are blocked.
        nrows (int, optional): Maximum number of rows to load from the CSV for memory safety. Default is 1000.

    Returns:
        Dict[str, Any]:
            - success (bool): Indicates whether the code executed successfully.
            - stdout (str): Any print output captured during execution.
            - result (Any): The result of the last expression in the code, converted to JSON-friendly format if possible.
            - error (str, optional): Error message if execution failed.
            - traceback (str, optional): Full traceback if an exception occurred.

    Safety Features:
        - Only allows specified safe modules (e.g., pandas, numpy) to be imported.
        - Blocks dangerous built-ins like exec, eval, open, compile, __import__.
        - Limits CSV rows to `nrows` to prevent memory overload.
        - Runs code in a restricted namespace with only `df`, `pd`, `np` and basic built-ins.

    Usage Notes:
        - Designed for the main LLM agent to perform computations like grouping, filtering,
          aggregation, or other pandas operations on user-uploaded CSVs.
        - Returns structured results suitable for further processing or display.
        - Ideal for interactive pipelines where the agent dynamically generates analysis code.

    Example:
        ```python
        result = run_pandas_code("sales.csv", "df.groupby('Category')['Sales'].sum().nlargest(10)")
        ```
    """
    try:
        abs_path = resolve_file_path(filename)
        if not os.path.exists(abs_path):
            return {"success": False, "error": f"File not found: {filename}"}

        # Load DataFrame with row cap
        df = safe_read_csv(abs_path, nrows=nrows)
        if df is None:
            return {"success": False, "error": "Failed to load CSV"}

        # AST Safety check
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in ALLOWED_MODULES:
                            return {"success": False, "error": f"Illegal import: {alias.name}"}
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ["exec", "eval", "open", "compile", "__import__"]:
                        return {"success": False, "error": f"Illegal function call: {node.func.id}"}
        except Exception as e:
            return {"success": False, "error": f"Code validation failed: {str(e)}"}

        # Prepare restricted execution environment
        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "min": min, "max": max, "sum": sum
            },
            "pd": pd,
            "np": np,
        }
        safe_locals = {"df": df}

        # Capture output
        stdout_buf = io.StringIO()
        result_obj = None
        try:
            exec(compile(tree, "<pandas_exec>", "exec"), safe_globals, safe_locals)
            # If last expression is evaluable, return it
            if isinstance(tree.body[-1], ast.Expr):
                result_obj = eval(
                    compile(ast.Expression(tree.body[-1].value), "<pandas_exec>", "eval"),
                    safe_globals,
                    safe_locals
                )
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

        # Convert result to JSON-friendly
        result_str = None
        if result_obj is not None:
            try:
                if hasattr(result_obj, "to_dict"):
                    result_str = result_obj.to_dict()
                elif hasattr(result_obj, "to_json"):
                    result_str = result_obj.to_json()
                else:
                    result_str = str(result_obj)
            except Exception:
                result_str = str(result_obj)

        return {
            "success": True,
            "stdout": stdout_buf.getvalue(),
            "result": result_str,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
@mcp.tool()
def run_plotting_code(filename: str, code: str, nrows: int = 1000) -> Dict[str, Any]:
    """
    Executes Python code for plotting data on a CSV file.

    Purpose:
        This tool enables the creation of data visualizations (e.g., charts, graphs)
        from a CSV file. The generated plot is saved as a temporary image file,
        and its path is returned. It can handle code that generates multiple plots.

    Args:
        filename (str): Name of the CSV file to use for plotting.
        code (str): Python code (matplotlib and/or seaborn focused) to execute. The code must
                    - Operate on a DataFrame named `df`
                    - Restricted to safe operations; imports and unsafe functions are blocked.
                    - Do not use plt.show() as this code is just to generate and save plots.
        nrows (int, optional): The number of rows to load from the CSV. Defaults to 1000.

    Returns:
        Dict[str, Any]:
            - success (bool): True if the plot was created and saved successfully.
            - image_paths (List[str]): A list of absolute paths to the saved plot images.
            - stdout (str): Any console output from the code execution.
            - error (str, optional): An error message if the execution failed.
    """
    try:
        abs_path = resolve_file_path(filename)
        if not os.path.exists(abs_path):
            return {"success": False, "error": f"File not found: {filename}"}

        df = safe_read_csv(abs_path, nrows=nrows)
        if df is None:
            return {"success": False, "error": "Failed to load CSV"}

        # AST Safety check
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in ALLOWED_MODULES:
                            return {"success": False, "error": f"Illegal import: {alias.name}"}
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ["exec", "eval", "open", "compile", "__import__"]:
                        return {"success": False, "error": f"Illegal function call: {node.func.id}"}
        except Exception as e:
            return {"success": False, "error": f"Code validation failed: {str(e)}"}

        safe_globals = {
            "__builtins__": {"print": print},
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns
        }
        safe_locals = {"df": df}
        stdout_buf = io.StringIO()
        image_paths = []

        try:
            # Execute the user-provided code
            exec(compile(tree, "<plotting_exec>", "exec"), safe_globals, safe_locals)
            
            # Save and close all open figures
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                plot_filename = f"{uuid.uuid4()}.png"
                plot_path = os.path.join(PLOT_DIRECTORY, plot_filename)
                plt.savefig(plot_path)
                print(f"Plot saved to {plot_path}")
                image_paths.append(plot_path)
            
            # Close all figures to prevent memory leaks and stacking plots
            plt.close('all')

        except Exception as e:
            plt.close('all') # Ensure plots are closed even on error
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        
        if not image_paths:
            return {"success": False, "error": "No plots were generated. The plotting code must call a plotting function (e.g., `plt.plot()`, `sns.barplot()`)."}

        return {
            "success": True,
            "image_paths": image_paths,
            "stdout": stdout_buf.getvalue(),
        }

    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
if __name__ == "__main__":
    # Run the MCP server
    mcp.run()