#!/bin/bash
# ============================================
# Babelon Notification Script (Discord)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load configuration
if [ -f "$PROJECT_DIR/.env.notify" ]; then
    source "$PROJECT_DIR/.env.notify"
fi

# Get VM name (auto-detect if not set)
get_vm_name() {
    if [ -n "$VM_NAME" ]; then
        echo "$VM_NAME"
    else
        hostname
    fi
}

# Get current timestamp
get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Send Discord notification
# Usage: send_discord "message" [color]
# Colors: success=3066993, warning=15105570, error=15158332, info=3447003
send_discord() {
    local message="$1"
    local color="${2:-3447003}"  # Default: info (blue)
    local vm_name=$(get_vm_name)
    local timestamp=$(get_timestamp)
    
    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        echo "Error: DISCORD_WEBHOOK_URL not set"
        return 1
    fi
    
    # Create JSON payload with embed
    local payload=$(cat <<EOF
{
    "embeds": [{
        "title": "🤖 Babelon Monitor",
        "description": "${message}",
        "color": ${color},
        "fields": [
            {"name": "🖥️ VM", "value": "${vm_name}", "inline": true},
            {"name": "🕐 Time", "value": "${timestamp}", "inline": true}
        ],
        "footer": {"text": "Babelon Translation Service"}
    }]
}
EOF
)
    
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" > /dev/null 2>&1
    
    return $?
}

# Send simple text message (no embed)
send_discord_simple() {
    local message="$1"
    local vm_name=$(get_vm_name)
    
    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        echo "Error: DISCORD_WEBHOOK_URL not set"
        return 1
    fi
    
    curl -s -X POST "$DISCORD_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"[${vm_name}] ${message}\"}" > /dev/null 2>&1
    
    return $?
}

# Notification types
notify_start_success() {
    local model="${1:-unknown}"
    send_discord "🚀 **服務啟動成功**\\n\\n✅ 狀態: 運行中\\n📦 模型: \`${model}\`" 3066993
}

notify_start_failed() {
    local error="${1:-unknown error}"
    send_discord "❌ **服務啟動失敗**\\n\\n🔴 錯誤: ${error}" 15158332
}

notify_shutdown() {
    # 使用快速發送，加短超時時間
    local vm_name=$(get_vm_name)
    local timestamp=$(get_timestamp)
    
    # 記錄到本地日誌作為備份
    echo "[$timestamp] [${vm_name}] SHUTDOWN notification sent" >> /tmp/babelon-notify.log
    
    # 發送 Discord 通知（5 秒超時）
    timeout 5 bash -c "curl -s -X POST '$DISCORD_WEBHOOK_URL' \
        -H 'Content-Type: application/json' \
        -d '{\"embeds\": [{\"title\": \"🤖 Babelon Monitor\", \"description\": \"⚠️ **服務即將關閉**\\\\n\\\\n🔄 狀態: 收到關機訊號\", \"color\": 15105570, \"fields\": [{\"name\": \"🖥️ VM\", \"value\": \"${vm_name}\", \"inline\": true}, {\"name\": \"🕐 Time\", \"value\": \"${timestamp}\", \"inline\": true}], \"footer\": {\"text\": \"Babelon Translation Service\"}}]}' \
        > /dev/null 2>&1" || echo "[$timestamp] Discord webhook timeout or failed" >> /tmp/babelon-notify.log
}

notify_health_ok() {
    local model="${1:-unknown}"
    send_discord "💚 **健康檢查通過**\\n\\n✅ 狀態: 健康\\n📦 模型: \`${model}\`" 3066993
}

notify_health_failed() {
    local error="${1:-unknown}"
    send_discord "🔴 **健康檢查失敗**\\n\\n❌ 錯誤: ${error}" 15158332
}

notify_build_test_start() {
    send_discord "🔧 **開始打包測試**\\n\\n⏳ 正在測試所有模型..." 3447003
}

notify_build_test_complete() {
    local report="$1"
    send_discord "📦 **打包測試完成**\\n\\n${report}" 3066993
}

notify_build_test_failed() {
    local report="$1"
    send_discord "❌ **打包測試失敗**\\n\\n${report}" 15158332
}

notify_custom() {
    local message="$1"
    local type="${2:-info}"  # success, warning, error, info
    
    case "$type" in
        success) send_discord "$message" 3066993 ;;
        warning) send_discord "$message" 15105570 ;;
        error)   send_discord "$message" 15158332 ;;
        *)       send_discord "$message" 3447003 ;;
    esac
}

# CLI interface
case "$1" in
    start-success)
        notify_start_success "$2"
        ;;
    start-failed)
        notify_start_failed "$2"
        ;;
    shutdown)
        notify_shutdown
        ;;
    health-ok)
        notify_health_ok "$2"
        ;;
    health-failed)
        notify_health_failed "$2"
        ;;
    build-test-start)
        notify_build_test_start
        ;;
    build-test-complete)
        notify_build_test_complete "$2"
        ;;
    build-test-failed)
        notify_build_test_failed "$2"
        ;;
    custom)
        notify_custom "$2" "$3"
        ;;
    test)
        echo "Testing Discord notification..."
        send_discord "🔔 **測試通知**\\n\\n這是一條測試訊息" 3447003
        echo "Done! Check your Discord channel."
        ;;
    *)
        echo "Usage: $0 {start-success|start-failed|shutdown|health-ok|health-failed|build-test-start|build-test-complete|build-test-failed|custom|test} [args]"
        exit 1
        ;;
esac
