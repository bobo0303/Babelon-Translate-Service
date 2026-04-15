#!/bin/bash
# ============================================
# Babelon Post-Build Test Script
# 打包後完整測試腳本
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTIFY_SCRIPT="$SCRIPT_DIR/notify.sh"

# Load configuration
source "$PROJECT_DIR/.env.notify"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVICE_URL="http://localhost:${SERVICE_PORT:-80}"
TEST_AUDIO="${TEST_AUDIO_FILE:-$PROJECT_DIR/audio/test.wav}"

# Models to test (from config or default)
IFS=',' read -ra MODELS <<< "${TEST_MODELS:-ggml_large_v2,ggml_large_v3,ggml_breeze_asr_25}"

# Results tracking
declare -A MODEL_LOAD_TIMES
declare -A MODEL_LOAD_RESULTS
declare -A MODEL_PIPELINE_RESULTS
AZURE_LID_RESULT="SKIP"
TOTAL_PASSED=0
TOTAL_FAILED=0
CURRENT_TRANSCRIPTION="unknown"
CURRENT_TRANSLATION="unknown"

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅${NC} $1"
}

log_error() {
    echo -e "${RED}❌${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Wait for service to be ready
wait_for_service() {
    local max_wait=${1:-120}
    local count=0
    log "等待服務就緒..."
    
    while [ $count -lt $max_wait ]; do
        if curl -s "$SERVICE_URL/health_check" | grep -q "OK"; then
            return 0
        fi
        sleep 2
        count=$((count + 2))
        echo -n "."
    done
    echo ""
    return 1
}

# Change transcription model
change_model() {
    local model_name="$1"
    local start_time=$(date +%s.%N)
    
    log "切換模型: $model_name"
    
    local response=$(curl -s -X POST "$SERVICE_URL/change_transcription_model" \
        -F "model_name=$model_name" 2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local load_time=$(echo "$end_time - $start_time" | bc)
    
    MODEL_LOAD_TIMES[$model_name]=$(printf "%.2f" $load_time)
    
    if echo "$response" | grep -q '"status":"OK"'; then
        MODEL_LOAD_RESULTS[$model_name]="PASS"
        log_success "$model_name 載入成功 (${MODEL_LOAD_TIMES[$model_name]}s)"
        return 0
    else
        MODEL_LOAD_RESULTS[$model_name]="FAIL"
        log_error "$model_name 載入失敗"
        return 1
    fi
}

# Test translate_pipeline
test_pipeline() {
    local model_name="$1"
    
    log "測試 translate_pipeline..."
    
    if [ ! -f "$TEST_AUDIO" ]; then
        log_error "測試音檔不存在: $TEST_AUDIO"
        MODEL_PIPELINE_RESULTS[$model_name]="FAIL"
        return 1
    fi
    
    local response=$(curl -s -X POST "$SERVICE_URL/translate_pipeline" \
        -F "file=@$TEST_AUDIO" \
        -F "meeting_id=test" \
        -F "device_id=test" \
        -F "audio_uid=test_$(date +%s)" \
        -F "times=$(date -Iseconds)" \
        -F "o_lang=zh" \
        -F "t_lang=en" \
        --max-time 120 2>/dev/null)
    
    if echo "$response" | grep -q '"status":"OK"'; then
        MODEL_PIPELINE_RESULTS[$model_name]="PASS"
        log_success "Pipeline 測試通過"
        return 0
    else
        MODEL_PIPELINE_RESULTS[$model_name]="FAIL"
        log_error "Pipeline 測試失敗"
        return 1
    fi
}

# Test Azure language detection
test_azure_lid() {
    log "測試 Azure 語言檢測..."
    
    if [ ! -f "$TEST_AUDIO" ]; then
        log_error "測試音檔不存在"
        AZURE_LID_RESULT="FAIL"
        return 1
    fi
    
    # 調用 /language_detect 端點，使用 method=azure_speech
    local lid_response=$(curl -s -X POST "$SERVICE_URL/language_detect" \
        -F "file=@$TEST_AUDIO" \
        -F "method=azure_speech" \
        --max-time 30 2>/dev/null)
    
    if echo "$lid_response" | grep -q '"status":"OK"'; then
        local detected_lang=$(echo "$lid_response" | grep -o '"detected_language":"[^"]*"' | cut -d'"' -f4)
        AZURE_LID_RESULT="PASS"
        log_success "Azure 語言檢測測試通過 (檢測到: $detected_lang)"
        return 0
    else
        local error_msg=$(echo "$lid_response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4 | head -1)
        AZURE_LID_RESULT="FAIL"
        log_error "Azure 語言檢測測試失敗: $error_msg"
        return 1
    fi
}

# Restart service to restore initial state
restart_service() {
    log "重啟服務恢復初始狀態..."
    
    cd "$PROJECT_DIR"
    docker compose restart > /dev/null 2>&1
    
    if wait_for_service 120; then
        log_success "服務已重啟並恢復初始狀態"
        return 0
    else
        log_error "服務重啟後未能正常啟動"
        return 1
    fi
}

# Get current models
get_current_models() {
    local response=$(curl -s "$SERVICE_URL/get_current_model" 2>/dev/null)
    
    CURRENT_TRANSCRIPTION=$(echo "$response" | grep -o '"transcription":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    CURRENT_TRANSLATION=$(echo "$response" | grep -o '"translation":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
}

# Generate report
generate_report() {
    local report=""
    
    # Model results
    for model in "${MODELS[@]}"; do
        local load_result="${MODEL_LOAD_RESULTS[$model]:-SKIP}"
        local pipeline_result="${MODEL_PIPELINE_RESULTS[$model]:-SKIP}"
        local load_time="${MODEL_LOAD_TIMES[$model]:-N/A}"
        
        local icon="❓"
        if [ "$load_result" = "PASS" ] && [ "$pipeline_result" = "PASS" ]; then
            icon="✅"
            ((TOTAL_PASSED++))
        elif [ "$load_result" = "FAIL" ] || [ "$pipeline_result" = "FAIL" ]; then
            icon="❌"
            ((TOTAL_FAILED++))
        fi
        
        report+="$icon \`$model\`: 載入 ${load_time}s | Pipeline ${pipeline_result}\\n"
    done
    
    # Azure LID result
    local azure_icon="❓"
    if [ "$AZURE_LID_RESULT" = "PASS" ]; then
        azure_icon="✅"
        ((TOTAL_PASSED++))
    elif [ "$AZURE_LID_RESULT" = "FAIL" ]; then
        azure_icon="❌"
        ((TOTAL_FAILED++))
    elif [ "$AZURE_LID_RESULT" = "SKIP" ]; then
        azure_icon="⏭️"
    fi
    report+="\\n$azure_icon Azure 語言檢測: $AZURE_LID_RESULT\\n"
    
    # Summary
    report+="\\n---\\n"
    report+="📊 總計: $TOTAL_PASSED 通過 / $TOTAL_FAILED 失敗\\n"
    
    # Current models after restart
    report+="\\n🎯 **當前模型狀態**\\n"
    report+="📝 轉譯模型: \`${CURRENT_TRANSCRIPTION:-unknown}\`\\n"
    report+="🌐 翻譯模型: \`${CURRENT_TRANSLATION:-unknown}\`"
    
    echo "$report"
}

# Main execution
main() {
    echo ""
    echo "============================================"
    echo " Babelon Post-Build Test"
    echo " 打包後完整測試"
    echo "============================================"
    echo ""
    
    # Send start notification
    bash "$NOTIFY_SCRIPT" build-test-start
    
    # Check if service is running
    log "檢查服務狀態..."
    if ! wait_for_service 30; then
        log_error "服務未運行，請先啟動服務"
        bash "$NOTIFY_SCRIPT" build-test-failed "服務未運行"
        exit 1
    fi
    log_success "服務運行中"
    
    # Check test audio
    if [ ! -f "$TEST_AUDIO" ]; then
        log_error "測試音檔不存在: $TEST_AUDIO"
        bash "$NOTIFY_SCRIPT" build-test-failed "測試音檔不存在"
        exit 1
    fi
    log_success "測試音檔就緒"
    
    echo ""
    echo "----------------------------------------"
    echo " 開始測試模型"
    echo "----------------------------------------"
    
    # Test each model
    for model in "${MODELS[@]}"; do
        echo ""
        log "========== 測試: $model =========="
        
        # Change model
        if change_model "$model"; then
            # Wait a bit for model to be fully ready
            sleep 2
            # Test pipeline
            test_pipeline "$model"
        fi
        
        echo ""
    done
    
    # Test Azure Language Detection
    echo ""
    echo "----------------------------------------"
    echo " 測試 Azure 服務"
    echo "----------------------------------------"
    test_azure_lid
    
    # Restart to restore initial state
    echo ""
    echo "----------------------------------------"
    echo " 重啟服務"
    echo "----------------------------------------"
    restart_service
    
    # Get current models after restart
    get_current_models
    log "當前轉譯模型: $CURRENT_TRANSCRIPTION"
    log "當前翻譯模型: $CURRENT_TRANSLATION"
    
    # Generate and send report
    echo ""
    echo "----------------------------------------"
    echo " 測試報告"
    echo "----------------------------------------"
    
    local report=$(generate_report)
    echo -e "$report"
    
    # Send notification
    if [ $TOTAL_FAILED -eq 0 ]; then
        report+="\\n\\n✅ 服務已重啟，恢復初始狀態"
        bash "$NOTIFY_SCRIPT" build-test-complete "$report"
        echo ""
        log_success "所有測試通過！"
    else
        report+="\\n\\n⚠️ 部分測試失敗，請檢查"
        bash "$NOTIFY_SCRIPT" build-test-failed "$report"
        echo ""
        log_error "有 $TOTAL_FAILED 項測試失敗"
    fi
    
    echo ""
    echo "============================================"
    echo " 測試完成"
    echo "============================================"
}

# Run
main "$@"
