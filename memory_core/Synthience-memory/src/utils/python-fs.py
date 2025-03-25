import os
from pathlib import Path
from typing import List

from logger import logger

async def ensure_directory(dir_path: str) -> None:
    try:
        normalized_path = os.path.normpath(dir_path)
        logger.debug("FileSystem", "Ensuring directory exists", {"path": normalized_path})
        os.makedirs(normalized_path, exist_ok=True)
        logger.debug("FileSystem", "Directory created/verified", {"path": normalized_path})
    except Exception as error:
        logger.error("FileSystem", "Failed to create directory", {"path": dir_path, "error": str(error)})
        raise error

async def write_file(file_path: str, content: str) -> None:
    try:
        normalized_path = os.path.normpath(file_path)
        logger.debug("FileSystem", "Starting file write", {"path": normalized_path})
        
        # Ensure directory exists
        dir_path = os.path.dirname(normalized_path)
        await ensure_directory(dir_path)
        
        # Write file
        with open(normalized_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Verify content
        with open(normalized_path, 'r', encoding='utf-8') as f:
            written_content = f.read()
            if written_content != content:
                raise Exception("File content verification failed")
        
        logger.debug("FileSystem", "File write successful", {"path": normalized_path})
    except Exception as error:
        logger.error("FileSystem", "Failed to write file", {"path": file_path, "error": str(error)})
        raise error

async def read_file(file_path: str) -> str:
    try:
        normalized_path = os.path.normpath(file_path)
        logger.debug("FileSystem", "Reading file", {"path": normalized_path})
        with open(normalized_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug("FileSystem", "File read successful", {"path": normalized_path})
        return content
    except Exception as error:
        logger.error("FileSystem", "Failed to read file", {"path": file_path, "error": str(error)})
        raise error

async def delete_file(file_path: str) -> None:
    try:
        normalized_path = os.path.normpath(file_path)
        logger.debug("FileSystem", "Deleting file", {"path": normalized_path})
        os.unlink(normalized_path)
        logger.debug("FileSystem", "File deletion successful", {"path": normalized_path})
    except Exception as error:
        logger.error("FileSystem", "Failed to delete file", {"path": file_path, "error": str(error)})
        raise error

async def list_files(dir_path: str) -> List[str]:
    try:
        normalized_path = os.path.normpath(dir_path)
        logger.debug("FileSystem", "Listing directory contents", {"path": normalized_path})
        files = os.listdir(normalized_path)
        logger.debug("FileSystem", "Directory listing successful", {"path": normalized_path, "fileCount": len(files)})
        return files
    except Exception as error:
        logger.error("FileSystem", "Failed to list directory", {"path": dir_path, "error": str(error)})
        raise error

async def file_exists(file_path: str) -> bool:
    try:
        normalized_path = os.path.normpath(file_path)
        exists = os.path.exists(normalized_path)
        logger.debug("FileSystem", f"File {'exists' if exists else 'does not exist'}", {"path": normalized_path})
        return exists
    except Exception:
        logger.debug("FileSystem", "File does not exist", {"path": file_path})
        return False
