import os
import shutil
from azure.storage.blob import BlobServiceClient
from abc import ABC, abstractmethod

class StorageInterface(ABC):

    @abstractmethod
    def save_file(self, file_path: str, content: bytes) -> str:
        pass

    @abstractmethod
    def load_file(self, file_path: str) -> bytes:
        pass

    @abstractmethod
    def list_files(self, directory: str) -> list[str]:
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        pass

    @abstractmethod
    def create_directory(self, directory: str) -> None:
        pass

    @abstractmethod
    def delete_directory(self, directory: str) -> None:
        pass

    @abstractmethod
    def upload(self, local_path: str, destination_path: str) -> None:
        pass

    @abstractmethod
    def append_file(self, file_path: str, content: bytes) -> None:
        pass

    @abstractmethod
    def get_modified_time(self, file_path: str) -> float:
        pass

    @abstractmethod
    def directory_exists(self, directory: str) -> bool:
        pass


class LocalStorage(StorageInterface):

    def save_file(self, file_path: str, content: bytes) -> str:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path

    def load_file(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()

    def list_files(self, directory: str) -> list[str]:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def file_exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    def delete_file(self, file_path: str) -> None:
        os.remove(file_path)

    def create_directory(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

    def delete_directory(self, directory: str) -> None:
        shutil.rmtree(directory)

    def upload(self, local_path: str, destination_path: str) -> None:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(local_path, destination_path)

    def append_file(self, file_path: str, content: bytes) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'ab') as f:
            f.write(content)

    def get_modified_time(self, file_path: str) -> float:
        return os.path.getmtime(file_path)

    def directory_exists(self, directory: str) -> bool:
        return self.file_exists(directory)


class BlobStorage(StorageInterface):
    """
        Writes to blob storage, using local disk as a cache

        TODO: Allow configuration of temp dir instead of just using the same paths in both local and remote
    """
    def __init__(self, connection_string: str, container_name: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.local_storage = LocalStorage()

    def download(self, file_path: str) -> bytes:
        blob_client = self.container_client.get_blob_client(file_path)
        return blob_client.download_blob().readall()

    def sync(self, file_path: str) -> None:
        if not self.local_storage.file_exists(file_path):
            print(f"DEBUG: missing local version of {file_path} - downloading")
            self.local_storage.save_file(file_path, self.download(file_path))
        else:
            local_timestamp = self.local_storage.get_modified_time(file_path)
            remote_timestamp = self.get_modified_time(file_path)
            if local_timestamp < remote_timestamp:
                # We always write remotely before writing locally, so we expect local_timestamp to be > remote timestamp
                print(f"DBEUG: local version of {file_path} out of date - downloading")
                self.local_storage.save_file(file_path, self.download(file_path))

    
    def save_file(self, file_path: str, content: bytes) -> str:
        blob_client = self.container_client.get_blob_client(file_path)
        blob_client.upload_blob(content, overwrite=True)
        self.local_storage.save_file(file_path, content)
        return file_path

    def load_file(self, file_path: str) -> bytes:
        self.sync(file_path)
        return self.local_storage.load_file(file_path)

    def list_files(self, directory: str) -> list[str]:
        return [blob.name for blob in self.container_client.list_blobs(name_starts_with=directory)]

    def file_exists(self, file_path: str) -> bool:
        blob_client = self.container_client.get_blob_client(file_path)
        return blob_client.exists()

    def delete_file(self, file_path: str) -> None:
        self.local_storage.delete_file(file_path)
        blob_client = self.container_client.get_blob_client(file_path)
        blob_client.delete_blob()

    def create_directory(self, directory: str) -> None:
        # Blob storage doesn't have directories, so only create it locally
        self.local_storage.create_directory(directory)

    def delete_directory(self, directory: str) -> None:
        self.local_storage.delete_directory(directory)
        blobs_to_delete = self.container_client.list_blobs(name_starts_with=directory)
        for blob in blobs_to_delete:
            self.container_client.delete_blob(blob.name)

    def upload(self, local_path: str, destination_path: str) -> None:
        with open(local_path, "rb") as data:
            blob_client = self.container_client.get_blob_client(destination_path)
            blob_client.upload_blob(data, overwrite=True)
        self.local_storage.upload(local_path, destination_path)

    def append_file(self, file_path: str, content: bytes) -> None:
        blob_client = self.container_client.get_blob_client(file_path)
        if not blob_client.exists():
            blob_client.create_append_blob()
        else:
            self.sync(file_path)

        blob_client.append_block(content)
        self.local_storage.append_file(file_path, content)

    def get_modified_time(self, file_path: str) -> float:
        blob_client = self.container_client.get_blob_client(file_path)
        properties = blob_client.get_blob_properties()
        # Convert the UTC datetime to a UNIX timestamp
        return properties.last_modified.timestamp()

    def directory_exists(self, directory: str) -> bool:
        blobs = self.container_client.list_blobs(name_starts_with=directory)
        return next(blobs, None) is not None


class StorageFactory:
    @staticmethod
    def get_storage() -> StorageInterface:
        storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
        if storage_type == 'local':
            return LocalStorage()
        elif storage_type == 'blob':
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
            if not connection_string or not container_name:
                raise ValueError("Azure Blob Storage connection string and container name must be set")
            return BlobStorage(connection_string, container_name)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

