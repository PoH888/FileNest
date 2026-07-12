from sqlalchemy import String

from sqlalchemy.orm import Mapped, mapped_column
# Mapped 用来标记：这个类属性不是普通属性，而是需要映射到数据库字段的 ORM 属性。

from .database import Base

class Workspace(Base): # 继承 FileNest 的 ORM 基类
    """告诉 SQLAlchemy：
    FileNest 有一种叫 Workspace 的数据库对象。"""
    __tablename__ = "workspaces" # 保存在 SQLite 的 workspaces 表

    id: Mapped[int] = mapped_column(primary_key=True)
#       ORM 映射属性；Python 中对应整数 int
#                     该字段是主键

    name: Mapped[str] = mapped_column(String, nullable=False)
#                                     数据库不允许这个字段保存 NULL

    root_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True, # 给 root_path 添加唯一约束，SQLite 会拒绝第二条记录
    )