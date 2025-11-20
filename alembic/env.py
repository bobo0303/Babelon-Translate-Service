import sys
from pathlib import Path
from logging.config import fileConfig
from wjy3.database import Base
from wjy3 import YmlUtils
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from lib.meeting_record import *
from alembic import context

sys.path.append(str(Path(__file__).resolve().parent))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

x_args = context.get_x_argument(as_dictionary=True)
table_prefix = x_args.get("prefix", "babelon")

app_config = YmlUtils("app.yml")
active = app_config.get_val("database.active")
active = active if active else "dev"
prefix = app_config.get_val(f"database.application.{active}.prefix")

if config.get_main_option("sqlalchemy.url") == "":
    dialect = app_config.get_val(f"database.application.{active}.dialect")
    server = app_config.get_val(f"database.application.{active}.server")
    port = app_config.get_val(f"database.application.{active}.port")
    database = app_config.get_val(f"database.application.{active}.database")
    username = app_config.get_val(f"database.application.{active}.username")
    password = app_config.get_val(f"database.application.{active}.password")
    driver = app_config.get_val(f"database.application.{active}.driver")
    sqlalchemy_url = f"{dialect}://{username}:{password}@{server}:{port}/{database}"
    if driver:
        sqlalchemy_url = f"{sqlalchemy_url}?driver={driver}"
    config.set_main_option("sqlalchemy.url", sqlalchemy_url)


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and not name.startswith(
        app_config.get_val(f"database.application.{active}.prefix")
    ):
        return False
    return True


# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata


# Debug 用：列出所有模型中收集到的 Table
if not Base.metadata.tables:
    print("❌ Base.metadata.tables 是空的！請確認有正確 import 所有 model")


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    version_table_name = f"{prefix}_alembic_version"

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            version_table=version_table_name,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
