"""
Heartbeat service for health check and service registration.

Provides two heartbeat mechanisms:
1. Active: On startup, GET other services' /heartbeat to notify them
2. Passive: Provide /heartbeat endpoint for others to check if alive
"""
import os
import socket
import asyncio
import datetime
import logging
from typing import Optional, List

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

##############################################################################
# Configuration
##############################################################################

def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


##############################################################################
# Models
##############################################################################

class HeartbeatResponse(BaseModel):
    """Passive heartbeat response - returned when others call /heartbeat"""
    ip: str                    # Local IP address
    port: int                  # Service port
    started_at: str            # Start time (ISO format)
    status: str                # Status: healthy, degraded, unhealthy
    uptime_seconds: float      # Uptime in seconds


##############################################################################
# Heartbeat Service
##############################################################################

class HeartbeatService:
    """
    Simple heartbeat service
    
    Usage:
        from lib.config.constant import BACKEND_IP_LIST
        
        heartbeat_service = HeartbeatService(
            backend_ips=BACKEND_IP_LIST,
            port=80
        )
        
        # On startup, notify all backends
        await heartbeat_service.notify_backends()
        
        # Generate heartbeat response
        response = heartbeat_service.get_heartbeat_response(is_processing=False)
    """
    
    def __init__(
        self,
        backend_ips: List[str] = None,
        port: int = 80,
        timezone = None
    ):
        """
        Initialize heartbeat service
        
        Args:
            backend_ips: List of backend IPs to notify
            port: This service's port
            timezone: Timezone object (pytz timezone)
        """
        self.backend_ips = [ip for ip in (backend_ips or []) if ip]  # Filter empty strings
        self.port = port
        self.timezone = timezone
        self.local_ip = get_local_ip()
        self.start_time: Optional[datetime.datetime] = None
        
    def set_start_time(self, start_time: datetime.datetime = None):
        """Set service start time"""
        if start_time:
            self.start_time = start_time
        elif self.timezone:
            self.start_time = datetime.datetime.now(self.timezone)
        else:
            self.start_time = datetime.datetime.now()
    
    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds"""
        if not self.start_time:
            return 0.0
        
        if self.timezone:
            now = datetime.datetime.now(self.timezone)
        else:
            now = datetime.datetime.now()
            
        return (now - self.start_time).total_seconds()
    
    async def notify_backends(self) -> dict:
        """
        On startup, GET other services' /heartbeat to notify them we're alive
        (and confirm they're alive too)
            
        Returns:
            dict: {ip: success_bool} Result for each backend
        """
        if not self.backend_ips:
            logger.info(" | BACKEND_IP_LIST is empty, skipping startup notification | ")
            return {}
        
        results = {}
        
        for backend_ip in self.backend_ips:
            url = f"http://{backend_ip}/heartbeat"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f" | Notified {backend_ip} successfully (they are alive) | ")
                        results[backend_ip] = True
                    else:
                        logger.warning(f" | Failed to reach {backend_ip}: status {response.status_code} | ")
                        results[backend_ip] = False
            except Exception as e:
                logger.warning(f" | Failed to reach {backend_ip}: {e} | ")
                results[backend_ip] = False
        
        return results
    
    def get_heartbeat_response(
        self,
        is_processing: bool = False,
        model_loaded: bool = True
    ) -> HeartbeatResponse:
        """
        Generate heartbeat response data
        
        Args:
            is_processing: Whether currently processing a task
            model_loaded: Whether the model is loaded
            
        Returns:
            HeartbeatResponse: Heartbeat response data
        """
        if not model_loaded:
            status = "unhealthy"
        elif is_processing:
            status = "degraded"
        else:
            status = "healthy"
        
        return HeartbeatResponse(
            ip=self.local_ip,
            port=self.port,
            started_at=self.start_time.isoformat() if self.start_time else "",
            status=status,
            uptime_seconds=round(self.get_uptime_seconds(), 2)
        )


##############################################################################
# Factory function
##############################################################################

def create_heartbeat_service(
    backend_ips: List[str] = None,
    port: int = None,
    timezone = None
) -> HeartbeatService:
    """
    工廠函數：創建 HeartbeatService
    
    Args:
        backend_ips: 要通知的後端 IP 列表 (從 BACKEND_IP_LIST 傳入)
        port: 服務端口 (default: 從 PORT 環境變數或 80)
        timezone: 時區物件
        
    Returns:
        HeartbeatService: 配置好的心跳服務實例
    """
    return HeartbeatService(
        backend_ips=backend_ips or [],
        port=port or int(os.environ.get("PORT", 80)),
        timezone=timezone
    )
