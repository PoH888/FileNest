from pathlib import Path

from sqlalchemy import create_engine

from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from collections.abc import Iterator
# Iterator 表示这个函数会逐步交出一个值(这里交出的值是 Session)

class Base(DeclarativeBase):
    """哪些 ORM 模型属于 FileNest"""
    pass


# 数据库在哪里
DATABASE_PATH = Path(__file__).resolve().parent.parent / "filenest.db"
#                   当前位置    绝对路径    从 backend/app/ 返回到 backend/，最终位置：backend/filenest.db
# D:\Project\FileNest\backend\filenest.db

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"
# sqlite:///：告诉 SQLAlchemy 使用磁盘上的 SQLite 文件，
# as_posix()：把 Path 对象转换成使用正斜杠 / 的字符串
# sqlite:///D:/Project/FileNest/backend/filenest.db

# 怎样连接数据库
engine = create_engine(DATABASE_URL)
# create_engine()：创建数据库连接入口，但采用延迟连接，此时不一定已经生成数据库文件


# 建立 Session 工厂
# 管理当前这一次业务操作中的查询、修改和事务
# 请求 A → Session A
SessionFactory = sessionmaker(
    bind=engine, # 把 Session 工厂绑定到现有 engine
    expire_on_commit=False, # 提交成功后，已经加载的 id、name 和 root_path 可以继续使用
)

# 交出一个 Session
def get_session() -> Iterator[Session]:
    """为一次请求提供数据库 Session，并在使用结束后自动关闭。"""

    with SessionFactory() as session:
    # 调用 SessionFactory() 创建 Session，使用 with 管理它的生命周期
        yield session
        # return：交出结果后函数直接结束
        # yield：交出 Session 后暂停，之后还能回来执行清理
