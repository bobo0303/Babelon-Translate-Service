#!/bin/bash

# ============================================
# Babelon Translation Service - Start
# - 第一次部署/新 Build → 完整測試
# - 後續啟動 → 只發健康報告
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTIFY_SCRIPT="$SCRIPT_DIR/notify.sh"
POST_BUILD_TEST="$SCRIPT_DIR/post_build_test.sh"
IMAGE_ID_FILE="$PROJECT_DIR/.last_image_id"

# Load notification config early for error handler
if [ -f "$PROJECT_DIR/.env.notify" ]; then
    source "$PROJECT_DIR/.env.notify"
fi

# Error handler - send notification on any failure
on_error() {
    local exit_code=$?
    local line_no=$1
    echo -e "\033[0;31m❌ 腳本在第 ${line_no} 行失敗 (exit code: ${exit_code})\033[0m"
    
    if [ "$NOTIFY_ON_START" = "true" ] && [ -f "$NOTIFY_SCRIPT" ]; then
        bash "$NOTIFY_SCRIPT" start-failed "Build/啟動失敗 (line ${line_no}, exit ${exit_code})"
    fi
    exit $exit_code
}
trap 'on_error ${LINENO}' ERR

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Change to project directory
cd "$PROJECT_DIR"

echo "Starting Babelon Translation Service..."

# Check .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo -e "${YELLOW}No .env found. Creating from .env.example...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env with your configuration.${NC}"
    else
        echo -e "${RED}.env and .env.example not found.${NC}"
        exit 1
    fi
fi

# Create runtime directories
mkdir -p audio logs models

# ============================================
# 自動安裝 systemd 監控服務（VM 開關機通知）
# ============================================
install_monitor_service() {
    local INSTALL_SCRIPT="$SCRIPT_DIR/install_monitor.sh"
    
    # 檢查是否已安裝
    if systemctl list-unit-files 2>/dev/null | grep -q "babelon-monitor.service"; then
        return 0  # 已安裝，跳過
    fi
    
    # 檢查安裝腳本是否存在
    if [ ! -f "$INSTALL_SCRIPT" ]; then
        return 0  # 腳本不存在，跳過
    fi
    
    echo ""
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW} 檢測到尚未安裝 VM 關機通知服務${NC}"
    echo -e "${YELLOW}============================================${NC}"
    
    # 檢查是否有 sudo 權限
    if sudo -n true 2>/dev/null; then
        echo -e "${BLUE}自動安裝 systemd 監控服務...${NC}"
        sudo bash "$INSTALL_SCRIPT"
    else
        echo -e "${YELLOW}需要 sudo 權限來安裝 VM 關機通知服務${NC}"
        echo -e "${YELLOW}請手動執行: sudo ./scripts/install_monitor.sh${NC}"
    fi
    echo ""
}

# 嘗試安裝監控服務
install_monitor_service

# Get current image ID before build
OLD_IMAGE_ID=$(docker images -q babelon-translate-service:latest 2>/dev/null || echo "")

# Build & start
echo "Building and starting container..."
docker compose up -d --build

# Get new image ID after build
NEW_IMAGE_ID=$(docker images -q babelon-translate-service:latest 2>/dev/null || echo "")

# Check if this is a new build
IS_NEW_BUILD=false
if [ -z "$OLD_IMAGE_ID" ]; then
    # First time build
    IS_NEW_BUILD=true
    echo -e "${BLUE}檢測到: 首次部署${NC}"
elif [ "$OLD_IMAGE_ID" != "$NEW_IMAGE_ID" ]; then
    # Image changed
    IS_NEW_BUILD=true
    echo -e "${BLUE}檢測到: 新的 Build (Image 已更新)${NC}"
elif [ ! -f "$IMAGE_ID_FILE" ]; then
    # No record of previous successful test
    IS_NEW_BUILD=true
    echo -e "${BLUE}檢測到: 尚未執行過完整測試${NC}"
else
    # Check if recorded ID matches
    LAST_TESTED_ID=$(cat "$IMAGE_ID_FILE" 2>/dev/null || echo "")
    if [ "$LAST_TESTED_ID" != "$NEW_IMAGE_ID" ]; then
        IS_NEW_BUILD=true
        echo -e "${BLUE}檢測到: Image 與上次測試不同${NC}"
    fi
fi

echo ""
echo "Waiting for service to be ready..."

# Wait for service to be ready (max 3 minutes)
MAX_WAIT=180
COUNT=0
SERVICE_URL="http://localhost:${PORT:-80}"

while [ $COUNT -lt $MAX_WAIT ]; do
    if curl -s "$SERVICE_URL/health_check" 2>/dev/null | grep -q "OK"; then
        # Get current model
        MODEL=$(curl -s "$SERVICE_URL/get_current_model" 2>/dev/null | grep -o '"transcription":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        
        echo ""
        echo -e "${GREEN}Service started successfully!${NC}"
        echo "  URL:          $SERVICE_URL"
        echo "  Model:        $MODEL"
        echo "  Health check: $SERVICE_URL/health_check"
        echo "  Logs:         docker compose logs -f"
        echo "  Stop:         ./scripts/stop.sh"
        echo ""
        
        if [ "$IS_NEW_BUILD" = true ]; then
            # New build → Run full test
            echo -e "${BLUE}============================================${NC}"
            echo -e "${BLUE} 執行完整測試 (新部署)${NC}"
            echo -e "${BLUE}============================================${NC}"
            echo ""
            
            if [ -f "$POST_BUILD_TEST" ]; then
                bash "$POST_BUILD_TEST"
                
                # Record successful test
                echo "$NEW_IMAGE_ID" > "$IMAGE_ID_FILE"
                echo -e "${GREEN}完整測試完成，已記錄 Image ID${NC}"
            else
                echo -e "${RED}找不到測試腳本: $POST_BUILD_TEST${NC}"
                # Still send start notification
                if [ "$NOTIFY_ON_START" = "true" ] && [ -f "$NOTIFY_SCRIPT" ]; then
                    bash "$NOTIFY_SCRIPT" start-success "$MODEL"
                fi
            fi
        else
            # Not new build → Just health report
            echo -e "${GREEN}非新部署，只發送健康報告${NC}"
            if [ "$NOTIFY_ON_START" = "true" ] && [ -f "$NOTIFY_SCRIPT" ]; then
                bash "$NOTIFY_SCRIPT" start-success "$MODEL"
            fi
        fi
        
        exit 0
    fi
    
    sleep 3
    COUNT=$((COUNT + 3))
    echo -n "."
done

echo ""
echo -e "${RED}Service failed to start within $MAX_WAIT seconds.${NC}"
echo "Check logs: docker compose logs"

# Send failure notification
if [ "$NOTIFY_ON_START" = "true" ] && [ -f "$NOTIFY_SCRIPT" ]; then
    bash "$NOTIFY_SCRIPT" start-failed "啟動逾時 ($MAX_WAIT 秒)"
fi

exit 1
