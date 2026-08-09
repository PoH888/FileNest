# FileNest 第 35 课：规模证据架构决定（E35-05）

## 决定

1. **当前继续使用 SQLite。**
2. **本课不新增数据库索引。**
3. **本课不迁移 PostgreSQL。**
4. 后续若优化，优先分析扫描同步和文档索引的批处理/对象生命周期，而不是先更换数据库。

## 决定所依据的证据

- large：10000 个文件、8000 个文档、16000 个 Chunk。
- 文件扫描与 FileEntry 持久化中位数：14742.750 ms。
- 固定文件搜索中位数：3.951 ms。
- 离线 Fake Agent 查询中位数：3.945 ms，所有重复均 completed。
- 文档索引中位数：26201.679 ms。
- Python 峰值分配中位数：80.98 MiB。
- SQLite 主库及旁车文件中位数：25.95 MiB。
- 正式固定测量没有观察到数据库锁、写入失败或查询失败。

## 为什么现在不新增索引

- 当前文件搜索使用工作区过滤加 `icontains` 子串匹配；在 10,000 个文件下中位数仍低于 4 ms，没有形成测量瓶颈。
- `file_entries` 已有 `(workspace_id, relative_path)` 唯一约束，文档和 Chunk 也已有身份/关联索引；没有证据支持再增加普通 B-tree 索引。
- 以 `%keyword%` 形式进行的前置通配符子串查询通常不能直接从普通 B-tree 获益；在没有查询计划和退化证据时新增索引只会增加写入与存储成本。
- 规模化 `knowledge_search`/全文检索未在 E35-02 中测量，因此本决定不宣称文档全文检索无需专门索引；该问题需要独立证据。

## 为什么现在不迁移 PostgreSQL

- 当前证据来自本地、单用户、临时数据库工作负载；SQLite 数据量约 25.95 MiB，搜索仍为毫秒级，且没有锁冲突或并发失败。
- 当前最重阶段是文件系统扫描和 Python 文档解析/切分/ORM 持久化。仅更换数据库不能直接消除这些成本。
- 尚未测量多用户并发写入、远程服务部署、连接池压力或锁竞争；没有足够证据承担迁移复杂度。

## 重新评估条件

出现以下任一证据时，重新评估索引或 PostgreSQL：

- 代表性并发测试稳定复现 `database is locked`、写吞吐不足或事务等待；
- 规模化文件搜索或 `knowledge_search` 超出以后明确制定的产品延迟目标；
- 查询计划证明具体 SQL 可以从一个明确索引获益，并通过新增索引前后对照验证；
- 产品从本地单用户应用转为多用户远程服务，出现独立数据库服务、备份、权限或高可用需求；
- SQLite 文件大小、迁移时间或维护成本成为可重复的实际问题。

## 当前不做的事

- 不新增 Alembic migration。
- 不修改 SQLAlchemy engine 或数据库 URL。
- 不引入 PostgreSQL 驱动或服务依赖。
- 不实现未经规模查询证据支持的 FTS、额外 B-tree 或向量数据库。

## 证据文件 SHA-256

- `scale_measurements_e35-02.json`：`a4800491e0d642839ed7aaf917a6c7e9c6e5b40259ab79d58b19c90f20812e0d`
- `scale_measurements_e35-03.json`：`483c18bfa5de4d433a11e7ef8c99cd1287cd8955394dec5bd4ba460641a27a29`
- `scale_measurements_e35-04.md`：`f6c3473ed40629b29707b5fb606eb3606042c4d54083e37d578f791a254506a7`

## 结论

现有证据支持“继续 SQLite、暂不新增索引、暂不迁移 PostgreSQL”。这是当前规模和工作负载下的可撤销决定，不是永久承诺；后续必须由新的失败、查询计划或并发数据触发复审。

## 复现决定报告

```powershell
.\.venv\Scripts\python.exe -m backend.app.scale_architecture_decision `
  --scan-report backend\evaluation\scale_measurements_e35-02.json `
  --document-report backend\evaluation\scale_measurements_e35-03.json `
  --bottleneck-report backend\evaluation\scale_measurements_e35-04.md `
  --output backend\evaluation\scale_architecture_decision_e35-05-rerun.md
```
