"""Blob storage for raw financial documents with compliance support."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
import structlog
import aiofiles
import hashlib
from datetime import datetime

logger = structlog.get_logger()


class BlobStore(ABC):
    """Abstract base class for blob storage backends."""
    
    @abstractmethod
    async def upload_file(
        self, 
        file_path: Path, 
        blob_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Upload a file and return the blob URL/path."""
        pass
    
    @abstractmethod
    async def download_file(self, blob_name: str, local_path: Path) -> None:
        """Download a file from blob storage."""
        pass
    
    @abstractmethod
    async def delete_file(self, blob_name: str) -> None:
        """Delete a file from blob storage."""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in blob storage with metadata."""
        pass


class LocalBlobStore(BlobStore):
    """Local filesystem implementation of blob storage."""
    
    def __init__(self, base_path: str = "./data/documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(store_type="local_blob")
    
    async def upload_file(
        self, 
        file_path: Path, 
        blob_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Upload file to local storage."""
        try:
            destination = self.base_path / blob_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            async with aiofiles.open(file_path, 'rb') as src:
                async with aiofiles.open(destination, 'wb') as dst:
                    content = await src.read()
                    await dst.write(content)
            
            # Store metadata if provided
            if metadata:
                metadata_path = destination.with_suffix(destination.suffix + '.meta')
                async with aiofiles.open(metadata_path, 'w') as meta_file:
                    import json
                    await meta_file.write(json.dumps(metadata, indent=2, default=str))
            
            self.logger.info("File uploaded to local storage", blob_name=blob_name)
            return str(destination)
            
        except Exception as e:
            self.logger.error("Failed to upload file", blob_name=blob_name, error=str(e))
            raise
    
    async def download_file(self, blob_name: str, local_path: Path) -> None:
        """Download file from local storage (essentially a copy operation)."""
        try:
            source = self.base_path / blob_name
            
            if not source.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(source, 'rb') as src:
                async with aiofiles.open(local_path, 'wb') as dst:
                    content = await src.read()
                    await dst.write(content)
            
            self.logger.info("File downloaded from local storage", blob_name=blob_name)
            
        except Exception as e:
            self.logger.error("Failed to download file", blob_name=blob_name, error=str(e))
            raise
    
    async def delete_file(self, blob_name: str) -> None:
        """Delete file from local storage."""
        try:
            file_path = self.base_path / blob_name
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            
            if file_path.exists():
                file_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            self.logger.info("File deleted from local storage", blob_name=blob_name)
            
        except Exception as e:
            self.logger.error("Failed to delete file", blob_name=blob_name, error=str(e))
            raise
    
    async def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in local storage."""
        try:
            files = []
            search_path = self.base_path
            
            if prefix:
                search_path = self.base_path / prefix
            
            for file_path in search_path.rglob('*'):
                if file_path.is_file() and not file_path.name.endswith('.meta'):
                    # Get file stats
                    stat = file_path.stat()
                    
                    # Load metadata if available
                    metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
                    metadata = {}
                    if metadata_path.exists():
                        try:
                            async with aiofiles.open(metadata_path, 'r') as meta_file:
                                import json
                                content = await meta_file.read()
                                metadata = json.loads(content)
                        except:
                            pass
                    
                    # Calculate file hash for integrity
                    file_hash = await self._calculate_file_hash(file_path)
                    
                    files.append({
                        "name": str(file_path.relative_to(self.base_path)),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "hash": file_hash,
                        "metadata": metadata
                    })
            
            return files
            
        except Exception as e:
            self.logger.error("Failed to list files", error=str(e))
            raise
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()


class AzureBlobStore(BlobStore):
    """Azure Blob Storage implementation (placeholder)."""
    
    def __init__(self, connection_string: str, container_name: str = "financial-documents"):
        self.connection_string = connection_string
        self.container_name = container_name
        self.logger = logger.bind(store_type="azure_blob")
    
    async def upload_file(self, file_path: Path, blob_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Placeholder for Azure implementation
        raise NotImplementedError("Azure Blob Storage implementation coming soon")
    
    async def download_file(self, blob_name: str, local_path: Path) -> None:
        raise NotImplementedError("Azure Blob Storage implementation coming soon")
    
    async def delete_file(self, blob_name: str) -> None:
        raise NotImplementedError("Azure Blob Storage implementation coming soon")
    
    async def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("Azure Blob Storage implementation coming soon")


# Factory function for blob store creation
async def create_blob_store(store_type: str, **kwargs) -> BlobStore:
    """Factory function to create blob store instances."""
    
    if store_type.lower() == "local":
        return LocalBlobStore(**kwargs)
    elif store_type.lower() == "azure":
        return AzureBlobStore(**kwargs)
    elif store_type.lower() == "aws":
        # Placeholder for AWS S3 implementation
        raise NotImplementedError("AWS S3 implementation coming soon")
    else:
        raise ValueError(f"Unsupported blob store type: {store_type}")
