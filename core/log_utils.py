"""
Logging configuration for multi-agent oncology feature extraction.
"""

import os
import sys
import logging
from datetime import datetime as dt

# Global variables to store logger configuration
_logger = None
_log_filename = None


def setup_logging(script_name="default", log_dir=None):
    """
    Setup logging for a specific script.

    Args:
        script_name: Name of the script (e.g., "feature_definition", "extract_validate", "aggregator")
        log_dir: Optional custom log directory. If None, defaults to "logs"

    Returns:
        tuple: (logger, log_filename)
    """
    global _logger, _log_filename

    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log filename with timestamp and script name
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    _log_filename = os.path.join(log_dir, f'{script_name}_log_{timestamp}.txt')

    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(_log_filename, mode='a'),  # Append to file
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )

    _logger = logging.getLogger(__name__)

    # Log the start of the program
    _logger.info(f"Starting {script_name} - Log file: {_log_filename}")
    _logger.info("=" * 80)

    return _logger, _log_filename


def get_logger():
    """Get the configured logger (must call setup_logging first)."""
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    return _logger


def get_log_filename():
    """Get the current log filename."""
    if _log_filename is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    return _log_filename


# Default logger for backward compatibility (will be overridden by setup_logging)
logger = logging.getLogger(__name__)


def print(*args, **kwargs):
    """Override print to also log to file"""
    # Use _logger if available, otherwise use root logger (configured by setup_run_logging)
    if _logger is not None:
        current_logger = _logger
    else:
        current_logger = logging.getLogger()

    message = ' '.join(str(arg) for arg in args)
    current_logger.info(message)


# Keep original print available if needed
import builtins
original_print = builtins.print


# ============================================================================
# Detailed Logging Functions
# ============================================================================

def log_detailed_output(content, prefix, chunk_idx=None, batch_idx=None, iteration=None):
    """
    Log detailed LLM output to a separate file.

    Args:
        content: The content to log (string or dict with 'text' and 'thinking' keys)
        prefix: Type of content (e.g., "prompt", "response", "features")
        chunk_idx: Optional chunk index
        batch_idx: Optional batch index
        iteration: Optional iteration number

    Returns:
        str: Path to the detailed log file
    """
    from config.SNOW_config import DETAILED_LOGGING, DETAILED_LOG_DIR

    if not DETAILED_LOGGING:
        return None

    # Create detailed log directory
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    # Build filename
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename_parts = [prefix]

    if chunk_idx is not None:
        filename_parts.append(f"chunk{chunk_idx}")
    if batch_idx is not None:
        filename_parts.append(f"batch{batch_idx}")
    if iteration is not None:
        filename_parts.append(f"iter{iteration}")

    filename_parts.append(f"{timestamp}.txt")
    filename = "_".join(filename_parts)

    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    # Write content to file
    with open(filepath, 'w', encoding='utf-8') as f:
        # Handle both dict (full response) and string (legacy) formats
        if isinstance(content, dict):
            if content.get('thinking'):
                f.write("THINKING TRACE:\n")
                f.write("=" * 80 + "\n")
                f.write(content['thinking'])
                f.write("\n\n")

            f.write("RESPONSE:\n")
            f.write("=" * 80 + "\n")
            f.write(content.get('text', str(content)))
        else:
            f.write(content)

    return filepath


def log_align_features_query(response, chunk_idx, batch_idx):
    """
    Log a align_features query response to detailed log file (one file per chunk).

    Args:
        response: The LLM response (string or dict with 'text' and 'thinking' keys)
        chunk_idx: Chunk index
        batch_idx: Batch index within the chunk

    Returns:
        str: Path to response file
    """
    from config.SNOW_config import DETAILED_LOGGING, DETAILED_LOG_DIR

    if not DETAILED_LOGGING:
        return None

    # Create detailed log directory
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    # One file per chunk - append batches to the same file
    filename = f"align_response_chunk{chunk_idx}.txt"
    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    # Append response with batch separator and timestamp
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Batch {batch_idx} - {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        # Handle both dict (full response) and string (legacy) formats
        if isinstance(response, dict):
            if response.get('thinking'):
                f.write("THINKING TRACE:\n")
                f.write("-" * 80 + "\n")
                f.write(response['thinking'])
                f.write("\n\n")

            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response.get('text', str(response)))
        else:
            f.write(response)

        f.write("\n\n")

    return filepath


def log_merge_features_query(prompt, response, num_chunks):
    """
    Log a merge_features query (prompt and response) to detailed log file.

    Args:
        prompt: The prompt sent to the LLM
        response: The LLM response (string or dict with 'text' and 'thinking' keys)
        num_chunks: Number of chunks being unified

    Returns:
        str: Path to response file
    """
    from config.SNOW_config import DETAILED_LOGGING, DETAILED_LOG_DIR

    if not DETAILED_LOGGING:
        return None

    # Create detailed log directory
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    # Single file for merge operation
    filename = f"merge_response.txt"
    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    # Write prompt and response with timestamp
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Merge Features (combining {num_chunks} chunks) - {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        # Write prompt
        f.write("PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt)
        f.write("\n\n")

        # Handle both dict (full response) and string (legacy) formats
        if isinstance(response, dict):
            if response.get('thinking'):
                f.write("THINKING TRACE:\n")
                f.write("-" * 80 + "\n")
                f.write(response['thinking'])
                f.write("\n\n")

            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response.get('text', str(response)))
        else:
            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response)

        f.write("\n\n")

    return filepath


def log_aggregation_code_query(prompt, response, feature_name):
    """
    Log an aggregation code generation query (prompt and response) to detailed log file.

    Args:
        prompt: The prompt sent to the LLM
        response: The LLM response (string or dict with 'text' and 'thinking' keys)
        feature_name: Name of the aggregated feature

    Returns:
        str: Path to response file
    """
    from config.SNOW_config import DETAILED_LOGGING, DETAILED_LOG_DIR

    if not DETAILED_LOGGING:
        return None

    # Create detailed log directory
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    # Sanitize feature name for filename
    safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
    filename = f"aggregation_code_{safe_feature_name}.txt"
    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    # Write prompt and response with timestamp
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Aggregation Code Generation: {feature_name} - {timestamp}\n")
        f.write(f"{'='*80}\n\n")

        # Write prompt
        f.write("PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt)
        f.write("\n\n")

        # Handle both dict (full response) and string (legacy) formats
        if isinstance(response, dict):
            if response.get('thinking'):
                f.write("THINKING TRACE:\n")
                f.write("-" * 80 + "\n")
                f.write(response['thinking'])
                f.write("\n\n")

            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response.get('text', str(response)))
        else:
            f.write("RESPONSE:\n")
            f.write("-" * 80 + "\n")
            f.write(response)

        f.write("\n\n")

    return filepath


def log_extraction_query(response, row_idx, iteration=None):
    """
    Log an extraction query response to detailed log file.

    Args:
        response: The LLM response
        row_idx: Row index being processed
        iteration: Optional iteration number for retries

    Returns:
        str: Path to response file
    """
    from config.SNOW_config import DETAILED_LOGGING

    if not DETAILED_LOGGING:
        return None

    # Use row_idx as a pseudo-chunk for organization
    response_file = log_detailed_output(
        response,
        "extract_response",
        chunk_idx=row_idx,
        iteration=iteration
    )

    return response_file


def log_validation_query(response, feature_name, iteration=None):
    """
    Log a validation query response to detailed log file.

    Args:
        response: The LLM response
        feature_name: Name of the feature being validated
        iteration: Optional iteration number for retries

    Returns:
        str: Path to response file
    """
    from config.SNOW_config import DETAILED_LOGGING

    if not DETAILED_LOGGING:
        return None

    # Use feature_name in the filename
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    response_filename = f"validate_response_{feature_name}_{timestamp}.txt"
    if iteration is not None:
        response_filename = f"validate_response_{feature_name}_iter{iteration}_{timestamp}.txt"

    from config.SNOW_config import DETAILED_LOG_DIR
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    response_file = os.path.join(DETAILED_LOG_DIR, response_filename)

    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(response)

    return response_file


def log_tool_use(tool_name, tool_args, tool_result, feature_name=None, iteration=None):
    """
    Log tool use during LLM query with tools.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
        tool_result: Result returned by the tool
        feature_name: Optional feature name for context
        iteration: Optional iteration number

    Returns:
        str: Path to log file
    """
    from config.SNOW_config import DETAILED_LOGGING

    if not DETAILED_LOGGING:
        return None

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # Build filename
    if feature_name:
        filename = f"tool_use_{feature_name}_{tool_name}_{timestamp}.txt"
    else:
        filename = f"tool_use_{tool_name}_{timestamp}.txt"

    if iteration is not None:
        filename = filename.replace(".txt", f"_iter{iteration}.txt")

    from config.SNOW_config import DETAILED_LOG_DIR
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TOOL USE LOG - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tool Name: {tool_name}\n\n")
        f.write(f"Arguments:\n{tool_args}\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"Result:\n{tool_result}\n")
        f.write("=" * 80 + "\n")

    return filepath


def log_tool_conversation(conversation_log, feature_name=None, iteration=None):
    """
    Log complete tool use conversation including LLM responses after tool use.

    Args:
        conversation_log: Complete conversation including tool calls and LLM responses
        feature_name: Optional feature name for context
        iteration: Optional iteration number for context

    Returns:
        str: Path to log file
    """
    from config.SNOW_config import DETAILED_LOGGING

    if not DETAILED_LOGGING:
        return None

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # Build filename
    filename_parts = ["tool_conversation"]

    if feature_name:
        # Sanitize feature name for filename
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
        filename_parts.append(safe_feature_name)

    if iteration is not None:
        filename_parts.append(f"iter{iteration}")

    filename_parts.append(f"{timestamp}.txt")
    filename = "_".join(filename_parts)

    from config.SNOW_config import DETAILED_LOG_DIR
    os.makedirs(DETAILED_LOG_DIR, exist_ok=True)

    filepath = os.path.join(DETAILED_LOG_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TOOL CONVERSATION LOG - {timestamp}\n")
        if feature_name:
            f.write(f"Feature: {feature_name}\n")
        if iteration is not None:
            f.write(f"Validation Iteration: {iteration}\n")
        f.write("=" * 80 + "\n\n")
        f.write(conversation_log)
        f.write("\n\n" + "=" * 80 + "\n")

    return filepath
