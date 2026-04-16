"""
Health check service for service registration and monitoring.

Provides two health check mechanisms:
1. Active: On startup, GET backend's /health_check to notify it
2. Passive: Provide /health_check endpoint for backend to check if alive
"""
import os
import datetime
from typing import Optional

import httpx
from pydantic import BaseModel
from lib.core.logging_config import get_logger, create_logger

# 主 logger（啟動成功等重要訊息）
logger = get_logger(__name__)  

# Health check 專用 logger（只寫檔案，不輸出 terminal）
hc_logger = create_logger(
    "health_check",
    log_file="logs/health_check.log",
    console_output=False
)

##############################################################################
# Models
##############################################################################

class HealthCheckResponse(BaseModel):
    """Passive health check response - returned when others call /health_check"""
    port: int                  # Service port
    started_at: str            # Start time (ISO format)
    status: str                # Status: healthy, degraded, unhealthy


##############################################################################
# Health Check Service
##############################################################################

class HealthCheckService:
    """
    Simple health check service
    
    Usage:
        from lib.config.constant import BACKEND_DOMAIN
        
        health_check_service = HealthCheckService(
            backend_domain=BACKEND_DOMAIN,
            port=80
        )
        
        # On startup, notify backend
        await health_check_service.notify_backend()
        
        # Generate health check response
        response = health_check_service.get_health_check_response(is_processing=False)
    """
    
    def __init__(
        self,
        backend_domain: str = None,
        port: int = 80,
        timezone = None
    ):
        """
        Initialize health check service
        
        Args:
            backend_domain: Backend domain to notify (from BACKEND_DOMAIN)
            port: This service's port
            timezone: Timezone object (pytz timezone)
        """
        self.backend_domain = backend_domain or ""
        self.port = port
        self.timezone = timezone
        self.start_time: Optional[datetime.datetime] = None
        
    def set_start_time(self, start_time: datetime.datetime = None):
        """Set service start time"""
        if start_time:
            self.start_time = start_time
        elif self.timezone:
            self.start_time = datetime.datetime.now(self.timezone)
        else:
            self.start_time = datetime.datetime.now()
    
    async def notify_backend(self) -> bool:
        """
        On startup, GET backend's /health_check to notify it we're alive
        (and confirm it's alive too)
        
        Expects wjy3 BaseResponse format with status field.
        Only status == "OK" is considered success.
            
        Returns:
            bool: True if backend responded with status "OK"
        """
        if not self.backend_domain:
            logger.info(" | BACKEND_DOMAIN is empty, skipping startup notification | ")
            return False
        
        url = f"{self.backend_domain.rstrip('/')}/health_check"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    hc_logger.warning(f" | Backend health check failed: HTTP {response.status_code} from {self.backend_domain} | ")
                    return False
                
                # Parse JSON response
                try:
                    response_data = response.json()
                    status = response_data.get("status", "")
                    
                    if status == "OK":
                        hc_logger.info(f" | Backend health check success: {self.backend_domain} is alive (status: OK) | ")
                        return True
                    else:
                        hc_logger.warning(f" | Backend health check failed: {self.backend_domain} returned status '{status}' (expected 'OK') | ")
                        return False
                        
                except Exception as parse_error:
                    hc_logger.warning(f" | Backend health check failed: Invalid JSON response from {self.backend_domain}: {parse_error} | ")
                    return False
                    
        except httpx.TimeoutException:
            hc_logger.warning(f" | Backend health check failed: Timeout connecting to {self.backend_domain} | ")
            return False
        except Exception as e:
            hc_logger.warning(f" | Backend health check failed: Error connecting to {self.backend_domain}: {e} | ")
            return False
    
    def get_health_check_response(
        self,
        is_processing: bool = False,
        model_loaded: bool = True,
        has_pending_tasks: bool = False
    ) -> HealthCheckResponse:
        """
        Generate health check response data
        
        Args:
            is_processing: Whether currently processing a task
            model_loaded: Whether the model is loaded
            has_pending_tasks: Whether there are tasks waiting in the queue
            
        Returns:
            HealthCheckResponse: Health check response data
        """
        if not model_loaded:
            status = "unhealthy"
        elif is_processing or has_pending_tasks:
            status = "degraded"
        else:
            status = "healthy"
        
        return HealthCheckResponse(
            port=self.port,
            started_at=self.start_time.isoformat() if self.start_time else "",
            status=status
        )


##############################################################################
# Factory function
##############################################################################

def create_health_check_service(
    backend_domain: str = None,
    port: int = None,
    timezone = None
) -> HealthCheckService:
    """
    工廠函數：創建 HealthCheckService
    
    Args:
        backend_domain: 要通知的後端域名 (從 BACKEND_DOMAIN 傳入)
        port: 服務端口 (default: 從 PORT 環境變數或 80)
        timezone: 時區物件
        
    Returns:
        HealthCheckService: 配置好的健康檢查服務實例
    """
    return HealthCheckService(
        backend_domain=backend_domain or "",
        port=port or int(os.environ.get("PORT", 80)),
        timezone=timezone
    )
