# FileNest 第 35 课：规模测试与瓶颈证据（E35-04）

## 证据范围

- 输入：`scale_measurements_e35-02.json` 和 `scale_measurements_e35-03.json`。
- 规模：small、medium、large；每档 3 次。
- 本报告只汇总测量事实和限制，不执行索引、PostgreSQL 或 SQLite 架构决策。

## 失败类型汇总

| 行动 | 运行数 | 失败数 | 证据 |
| --- | ---: | ---: | --- |
| `scan_workspace` 与 FileEntry 持久化 | 9 | 0 | 报告生成前的文件计数断言全部通过 |
| `services.search_files` | 9 | 0 | 三档命中总数与规模一致 |
| 只读 Agent 查询 | 9 | 0 | completed：9/9 |
| 文档解析、切分与持久化 | 9 | 0 | 文档数和 Chunk 数在重复测量中一致 |

本次正式测量没有观察到产品行动失败；失败类型汇总中的 0 表示这些已完成的固定场景未失败，不代表所有异常组合都已覆盖。

## 中位数与样本范围

| 规模 | 文件/文档/Chunk | 扫描 ms | 搜索 ms | Agent ms | 文档索引 ms | Python 峰值分配 | SQLite 大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 100/80/160 | 179.850 (176.494-183.033) | 2.498 | 2.528 | 249.522 (248.963-316.121) | 1.68 MiB | 408.00 KiB |
| medium | 1000/800/1600 | 1551.430 (1548.058-1564.938) | 2.584 | 2.801 | 2461.450 (2416.048-10201.385) | 7.66 MiB | 2.73 MiB |
| large | 10000/8000/16000 | 14742.750 (14644.923-14762.789) | 3.951 | 3.945 | 26201.679 (25439.039-80762.585) | 80.98 MiB | 25.95 MiB |

## 可复现瓶颈

1. **扫描与文件索引持久化随文件数近似线性增长。** 扫描中位数从 179.850 ms 增长到 14742.750 ms；大规模下它明显高于毫秒级搜索和离线 Agent 查询。
2. **文档索引是当前最重的测量阶段。** 8000 个文档、16000 个 Chunk 的索引中位数为 26201.679 ms；从 small 到 large 约扩大 105.0 倍。
3. **内存和 SQLite 大小随文档规模增长。** 大规模中位数分别为 80.98 MiB 和 25.95 MiB；这说明继续扩大规模会直接增加资源成本。

## 当前不构成瓶颈的行动

固定 `item-` 查询的搜索中位数为 3.951 ms；使用 `FakeModelClient` 的 Agent 查询中位数为 3.945 ms。它们在本次受控工作负载下没有表现为主要耗时。

## 波动与限制

- 文档索引存在明显冷启动/文件缓存波动：medium 最高样本为 10,201.385 ms，large 最高样本为 80,762.585 ms；保留全部样本，不把中位数当成唯一真相。
- Agent 查询使用离线 `FakeModelClient`，不包含真实模型网络延迟、供应商排队或 token 成本。
- 内存指标是 `tracemalloc` Python 峰值分配，不等同于完整进程 RSS。
- 这些数据用于定位测量到的规模瓶颈，不替代生产压测，也不提前决定索引、PostgreSQL 或继续 SQLite。

## 复现汇总

```powershell
.\.venv\Scripts\python.exe -m backend.app.scale_bottleneck_summary `
  --scan-report backend\evaluation\scale_measurements_e35-02.json `
  --document-report backend\evaluation\scale_measurements_e35-03.json `
  --output backend\evaluation\scale_measurements_e35-04-rerun.md
```
