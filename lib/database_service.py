import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from wjy3 import get_logger, YmlUtils, get_db_session

app_config = YmlUtils("app.yml")
active = app_config.get_val("database.active")
active = active if active else "dev"


class DatabaseService:

    def __init__(self):
        root_dir = Path(__file__).parent.parent
        self.logger = get_logger("dbLogger")
        self.script_location = root_dir / "alembic"
        self.alembic_cfg_path = root_dir / "alembic.ini"

    async def initialize_database(self) -> None:
        """Initialize the database by creating necessary tables."""
        try:
            await self.run_migrations()
        except Exception as e:
            self.logger.error(f" | Failed to run migrations: {e} | ")
            raise

    async def run_migrations(self) -> None:
        should_initialize_alembic = False
        session = get_db_session()
        # If the table does not exist it throws an error
        # so we need to catch it
        try:
            version_table_name = f"{active}_alembic_version"
            session.exec(text(f"SELECT * FROM {version_table_name}"))
        except Exception:  # noqa: BLE001
            self.logger.debug(" | Alembic not initialized | ")
            should_initialize_alembic = True
        await asyncio.to_thread(self._run_migrations, should_initialize_alembic)

    def _run_migrations(self, should_initialize_alembic: bool) -> None:
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(self.script_location))

        dialect = app_config.get_val(f"database.application.{active}.dialect")
        server = app_config.get_val(f"database.application.{active}.server")
        port = app_config.get_val(f"database.application.{active}.port")
        database = app_config.get_val(f"database.application.{active}.database")
        username = app_config.get_val(f"database.application.{active}.username")
        password = app_config.get_val(f"database.application.{active}.password")
        driver = app_config.get_val(f"database.application.{active}.driver")
        sqlalchemy_url = f"{dialect}://{username}:{password}@{server}:{port}/{database}"
        if driver:
            sqlalchemy_url += f"?driver={driver}"
        alembic_cfg.set_main_option("sqlalchemy.url", sqlalchemy_url)

        if should_initialize_alembic:
            try:
                self.init_alembic(alembic_cfg)
            except Exception as exc:
                msg = f"Error initializing alembic: {exc}"
                self.logger.exception(msg)
                raise RuntimeError(msg) from exc
        else:
            self.logger.debug(" | Alembic initialized | ")

    @staticmethod
    def init_alembic(alembic_cfg) -> None:
        logging.info("Initializing alembic")
        command.ensure_version(alembic_cfg)

        app_config = YmlUtils("app.yml")
        active = app_config.get_val("database.active") or "dev"
        prefix = (
            app_config.get_val(f"database.application.{active}.prefix") or "myagents_"
        )
        print(f"init_alembic() Using table prefix: {prefix}")

        # 傳入 prefix 給 migration script
        alembic_cfg.attributes["prefix"] = prefix
        command.upgrade(alembic_cfg, "head")
