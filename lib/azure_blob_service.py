"""
Azure Blob Storage Service
用於上傳音訊檔案到 Azure Blob Storage
"""

import os
import asyncio
from typing import Optional
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError

import logging
import logging.handlers

logger = logging.getLogger(__name__)  
  
# Configure logger settings (if not already configured)  
if not logger.handlers:  
    log_format = "%(asctime)s - %(message)s"  
    log_file = "logs/app.log"  
    logging.basicConfig(level=logging.INFO, format=log_format)  
  
    # Create file handler  
    file_handler = logging.handlers.RotatingFileHandler(  
        log_file, maxBytes=10*1024*1024, backupCount=5  
    )  
    file_handler.setFormatter(logging.Formatter(log_format))  
  
    # Create console handler  
    console_handler = logging.StreamHandler()  
    console_handler.setFormatter(logging.Formatter(log_format))  
  
    logger.addHandler(file_handler)  
    logger.addHandler(console_handler)  
  
logger.setLevel(logging.INFO)  
logger.propagate = False  

class AzureBlobService:
    """Azure Blob Storage 服務類別"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "",
    ):
        """
        初始化 Azure Blob Service

        Args:
            connection_string: Azure Storage 連接字串，如果為 None 則從環境變數讀取
            container_name: Container 名稱
        """
        self.connection_string = connection_string
        self.container_name = container_name

        if not self.connection_string:
            logger.warning(
                "Azure Storage connection string not configured. "
                "Audio files will not be uploaded to Azure Blob Storage."
            )
            self.blob_service_client = None
        else:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                self._ensure_container_exists()
                logger.info(
                    f"Azure Blob Service initialized with container: {container_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Service: {e}")
                self.blob_service_client = None

    def _ensure_container_exists(self):
        """確保 container 存在，如果不存在則創建"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
        except AzureError as e:
            logger.error(f"Error ensuring container exists: {e}")

    def upload_file(
        self,
        local_file_path: str,
        blob_name: Optional[str] = None,
        meeting_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        上傳檔案到 Azure Blob Storage

        Args:
            local_file_path: 本地檔案路徑
            blob_name: Blob 名稱，如果為 None 則使用檔案名稱
            meeting_id: 會議 ID，用於組織檔案結構

        Returns:
            Blob URL 如果成功，否則返回 None
        """
        if not self.blob_service_client:
            logger.warning("Azure Blob Service not configured, skipping upload")
            return None

        if not os.path.exists(local_file_path):
            logger.error(f"Local file not found: {local_file_path}")
            return None

        try:
            # 如果沒有指定 blob_name，使用檔案名稱
            if blob_name is None:
                blob_name = os.path.basename(local_file_path)

            # 如果有 meeting_id，加入路徑結構
            if meeting_id:
                blob_name = f"recordings/{meeting_id}/{blob_name}"

            # 取得 blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )

            # 設定 content type
            content_settings = ContentSettings(content_type="audio/wav")

            # 上傳檔案
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(
                    data, overwrite=True, content_settings=content_settings
                )

            blob_url = blob_client.url
            logger.info(f"Successfully uploaded file to Azure Blob: {blob_url}")
            return blob_url

        except AzureError as e:
            logger.error(f"Failed to upload file to Azure Blob: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Azure Blob upload: {e}")
            return None

    async def upload_file_async(
        self,
        local_file_path: str,
        blob_name: Optional[str] = None,
        meeting_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        異步上傳檔案到 Azure Blob Storage

        Args:
            local_file_path: 本地檔案路徑
            blob_name: Blob 名稱，如果為 None 則使用檔案名稱
            meeting_id: 會議 ID，用於組織檔案結構

        Returns:
            Blob URL 如果成功，否則返回 None
        """
        # 在 executor 中執行同步的上傳操作
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.upload_file, local_file_path, blob_name, meeting_id
        )

    def delete_file(self, blob_name: str, meeting_id: Optional[str] = None) -> bool:
        """
        從 Azure Blob Storage 刪除檔案

        Args:
            blob_name: Blob 名稱
            meeting_id: 會議 ID

        Returns:
            成功返回 True，否則返回 False
        """
        if not self.blob_service_client:
            logger.warning("Azure Blob Service not configured, skipping delete")
            return False

        try:
            # 如果有 meeting_id，加入路徑結構
            if meeting_id:
                blob_name = f"{meeting_id}/{blob_name}"

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Successfully deleted blob: {blob_name}")
            return True

        except AzureError as e:
            logger.error(f"Failed to delete blob: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during blob deletion: {e}")
            return False


# 全域實例
_azure_blob_service = None


def get_azure_blob_service(
    connection_string: Optional[str] = None, container_name: str = ""
) -> AzureBlobService:
    """
    取得 Azure Blob Service 單例

    Args:
        connection_string: Azure Storage 連接字串
        container_name: Container 名稱

    Returns:
        AzureBlobService 實例
    """
    global _azure_blob_service

    connection_string = connection_string or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    # print("connection_string", connection_string)
    container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    # print("container_name", container_name)

    if _azure_blob_service is None:
        _azure_blob_service = AzureBlobService(connection_string, container_name)
    return _azure_blob_service
