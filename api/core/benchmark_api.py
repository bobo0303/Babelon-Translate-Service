"""
Benchmark API Router

提供 Benchmark 測試的 API endpoints

使用方式:
    在 main.py 中:
    
    from api.core.benchmark_api import benchmark_router
    app.include_router(benchmark_router, prefix="/benchmark", tags=["Benchmark"])
"""

import os
from fastapi import APIRouter, Form
from typing import Optional

from wjy3 import BaseResponse, Status
from api.core.trim_benchmark_recorder import (
    get_benchmark_recorder,
    compare_results,
    print_comparison
)
from lib.config.constant import ENABLE_TRIM

benchmark_router = APIRouter()


@benchmark_router.post("/start")
async def start_benchmark(
    test_name: str = Form(..., description="測試名稱，用於生成報告文件名"),
    trim_enabled: bool = Form(None, description="是否啟用 trim（默認使用當前系統設定）")
):
    """
    開始 Benchmark 測試
    
    開始記錄所有 translate_pipeline API 的結果數據。
    
    Args:
        test_name: 測試名稱（如 "test_with_trim", "test_without_trim"）
        trim_enabled: 是否啟用 trim（如果不傳，使用當前系統的 ENABLE_TRIM 設定）
    """
    recorder = get_benchmark_recorder()
    
    # 如果已有正在進行的測試，先結束它
    if recorder.is_enabled():
        recorder.end_test()
    
    # 使用傳入的值，或者使用系統當前設定
    actual_trim_enabled = trim_enabled if trim_enabled is not None else ENABLE_TRIM
    
    recorder.start_test(test_name, trim_enabled=actual_trim_enabled)
    
    return BaseResponse(
        status=Status.OK,
        message=f" | Benchmark test '{test_name}' started | trim_enabled={actual_trim_enabled} | ",
        data={
            "test_name": test_name,
            "trim_enabled": actual_trim_enabled,
            "recording": True
        }
    )


@benchmark_router.post("/stop")
async def stop_benchmark(
    export_report: bool = Form(True, description="是否自動導出報告"),
    output_path: str = Form(None, description="報告輸出路徑（可選）")
):
    """
    停止 Benchmark 測試並生成報告
    
    停止記錄並計算統計數據，可選擇導出 JSON 報告。
    
    Returns:
        測試摘要統計
    """
    recorder = get_benchmark_recorder()
    
    if not recorder.is_enabled():
        return BaseResponse(
            status=Status.FAILED,
            message=" | No active benchmark test to stop | ",
            data=None
        )
    
    # 結束測試並計算統計
    stats = recorder.end_test()
    
    # 打印摘要
    recorder.print_summary()
    
    # 導出報告
    report_path = None
    if export_report:
        report_path = recorder.export_report(output_path)
    
    return BaseResponse(
        status=Status.OK,
        message=f" | Benchmark test stopped | Report: {report_path} | ",
        data={
            "test_name": stats.test_name if stats else None,
            "trim_enabled": stats.trim_enabled if stats else None,
            "total_requests": stats.total_requests if stats else 0,
            "total_cancelled": stats.total_cancelled if stats else 0,
            "cancel_rate": f"{stats.cancel_rate:.2%}" if stats else "0%",
            "avg_response_time": f"{stats.avg_response_time:.4f}s" if stats else "0s",
            "report_path": report_path
        }
    )


@benchmark_router.get("/status")
async def benchmark_status():
    """
    獲取當前 Benchmark 狀態
    
    Returns:
        是否正在記錄以及當前測試資訊
    """
    recorder = get_benchmark_recorder()
    
    if recorder.is_enabled():
        summary = recorder.get_summary()
        return BaseResponse(
            status=Status.OK,
            message=" | Benchmark is recording | ",
            data={
                "recording": True,
                **summary
            }
        )
    else:
        return BaseResponse(
            status=Status.OK,
            message=" | Benchmark is not recording | ",
            data={"recording": False}
        )


@benchmark_router.get("/summary")
async def get_benchmark_summary():
    """
    獲取當前測試的即時摘要
    
    Returns:
        當前測試的統計摘要
    """
    recorder = get_benchmark_recorder()
    summary = recorder.get_summary()
    
    if not summary:
        return BaseResponse(
            status=Status.FAILED,
            message=" | No benchmark data available | ",
            data=None
        )
    
    return BaseResponse(
        status=Status.OK,
        message=" | Benchmark summary | ",
        data=summary
    )


@benchmark_router.get("/final_texts")
async def get_final_texts():
    """
    獲取所有 UID 的最終文本
    
    Returns:
        Dict[audio_uid, final_text]
    """
    recorder = get_benchmark_recorder()
    final_texts = recorder.get_final_texts()
    
    return BaseResponse(
        status=Status.OK,
        message=f" | Total {len(final_texts)} UIDs | ",
        data=final_texts
    )


@benchmark_router.post("/compare")
async def compare_benchmark_reports(
    report_with_trim: str = Form(..., description="啟用 trim 的報告 JSON 路徑"),
    report_without_trim: str = Form(..., description="未啟用 trim 的報告 JSON 路徑"),
    output_path: str = Form(None, description="比較報告輸出路徑（可選）")
):
    """
    比較兩個 Benchmark 測試報告
    
    比較 trim 啟用 vs 未啟用 的測試結果，包括：
    - 響應時間比較
    - Cancel 率比較
    - 文本相似性分析
    - 連續 Cancel 分布比較
    
    Args:
        report_with_trim: 啟用 trim 的報告路徑
        report_without_trim: 未啟用 trim 的報告路徑
        output_path: 比較報告輸出路徑
    """
    # 檢查文件是否存在
    if not os.path.exists(report_with_trim):
        return BaseResponse(
            status=Status.FAILED,
            message=f" | Report file not found: {report_with_trim} | ",
            data=None
        )
    
    if not os.path.exists(report_without_trim):
        return BaseResponse(
            status=Status.FAILED,
            message=f" | Report file not found: {report_without_trim} | ",
            data=None
        )
    
    try:
        comparison = compare_results(report_with_trim, report_without_trim, output_path)
        print_comparison(comparison)
        
        return BaseResponse(
            status=Status.OK,
            message=" | Comparison completed | ",
            data=comparison
        )
    except Exception as e:
        return BaseResponse(
            status=Status.FAILED,
            message=f" | Comparison failed: {e} | ",
            data=None
        )


@benchmark_router.get("/list_reports")
async def list_benchmark_reports():
    """
    列出所有 benchmark 報告文件
    
    Returns:
        benchmark_results 目錄下的所有 JSON 報告
    """
    reports_dir = "benchmark_results"
    
    if not os.path.exists(reports_dir):
        return BaseResponse(
            status=Status.OK,
            message=" | No benchmark reports found | ",
            data=[]
        )
    
    reports = []
    for filename in os.listdir(reports_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(reports_dir, filename)
            stat = os.stat(filepath)
            reports.append({
                "filename": filename,
                "filepath": filepath,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
    
    # 按修改時間排序
    reports.sort(key=lambda x: x['modified'], reverse=True)
    
    return BaseResponse(
        status=Status.OK,
        message=f" | Found {len(reports)} benchmark reports | ",
        data=reports
    )
