#!/bin/bash
# ============================================
# Babelon Health Monitor Script
# 健康監控腳本 - 用於開機/重啟時回報狀態
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTIFY_SCRIPT="$SCRIPT_DIR/notify.sh"

# Load configuration
if [ -f "$PROJECT_DIR/.env.notify" ]; then
    source "$PROJECT_DIR/.env.notify"
fi

SERVICE_URL="http://localhost:${SERVICE_PORT:-80}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for service to be ready
wait_for_service() {
    local max_wait=${1:-180}  # 預設等待3分鐘
    local count=0
    
    log "等待服務就緒..."
    
    while [ $count -lt $max_wait ]; do
        local response=$(curl -s "$SERVICE_URL/health_check" 2>/dev/null)
        
        if echo "$response" | grep -q '"status":"OK"'; then
            return 0
        fi
        
        sleep 3
        count=$((count + 3))
        echo -n "."
    done
    
    echo ""
    return 1
}

# Get current model
get_current_model() {
    local response=$(curl -s "$SERVICE_URL/get_current_model" 2>/dev/null)
    
    if echo "$response" | grep -q "transcription"; then
        echo "$response" | grep -o '"transcription":"[^"]*"' | cut -d'"' -f4
    else
        echo "unknown"
    fi
}

# Health check
health_check() {
    log "執行健康檢查..."
    
    if wait_for_service 180; then
        local model=$(get_current_model)
        log "✅ 服務健康，模型: $model"
        bash "$NOTIFY_SCRIPT" health-ok "$model"
        return 0
    else
        log "❌ 服務未響應"
        bash "$NOTIFY_SCRIPT" health-failed "服務未響應健康檢查"
        return 1
    fi
}

# Startup notification (called after VM boot)
startup_notify() {
    log "系統啟動，檢查服務狀態..."
    
    # Wait a bit for Docker to start
    sleep 10
    
    # Check if container is running
    cd "$PROJECT_DIR"
    
    if docker compose ps 2>/dev/null | grep -q "Up"; then
        log "容器運行中，等待服務就緒..."
        health_check
    else
        log "容器未運行，嘗試啟動..."
        docker compose up -d
        
        if wait_for_service 180; then
            local model=$(get_current_model)
            log "✅ 服務啟動成功"
            bash "$NOTIFY_SCRIPT" start-success "$model"
        else
            log "❌ 服務啟動失敗"
            bash "$NOTIFY_SCRIPT" start-failed "容器啟動後服務未響應"
        fi
    fi
}

# Shutdown notification
shutdown_notify() {
    log "系統關機，發送通知..."
    bash "$NOTIFY_SCRIPT" shutdown
}

# Main
case "$1" in
    health)
        health_check
        ;;
    startup)
        startup_notify
        ;;
    shutdown)
        shutdown_notify
        ;;
    *)
        echo "Usage: $0 {health|startup|shutdown}"
        echo ""
        echo "  health   - 執行健康檢查並通知"
        echo "  startup  - 系統啟動時調用，檢查並啟動服務"
        echo "  shutdown - 系統關機前調用，發送關機通知"
        exit 1
        ;;
esac
