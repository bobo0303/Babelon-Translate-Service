import asyncio
from wjy3 import get_logger
from lib.database_service import DatabaseService

# === Logging ===
log = get_logger()


# === 主入口 ===
async def main():
    log.info("=== Database Initialization ===")
    try:
        await DatabaseService().initialize_database()
        log.info("=== Database Initialization Success ===")
    except Exception as e:
        log.exception("❌ Database Initialization Failed")
        raise e

    # await asyncio.Event().wait()  # 保持常駐


if __name__ == "__main__":
    print("=== myAgents Scheduler Starting ===")
    asyncio.run(main())
