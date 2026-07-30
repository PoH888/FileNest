# FileNest 第 26 课：安全整理 Agent 综合评测

## 评测边界

- 评测套件：`safe_organization_v1`
- 使用临时 SQLite 与 pytest 临时工作区，不接触真实用户文件。
- 本报告验证确定性的程序安全边界，不代表真实模型质量。
- 报告不保存绝对路径、用户文件内容、提示词或工具载荷。

## 评测源

- `tests/test_safe_organization_end_to_end.py` SHA-256：`9d2a427b0bf84f9500ff14f18f3fee5a455ba2168e9032a46a2f3d3d05c6f2fe`
- `tests/test_approval_disk_immutability.py` SHA-256：`06d257384941bb00ca54fda5268c171e4c881b551a4d2e4ef5fa496bc7c16e83`

## 汇总结果

| 指标 | 结果 |
| --- | ---: |
| 综合安全场景通过 | 9/9 |
| 安全整理主链通过 | 5/5 |
| 未经审批磁盘快照一致 | 4/4 |
| 未经审批磁盘变更 | 0 |

## 用例结果

| 用例 | 分类 | 结果 |
| --- | --- | --- |
| `test_query_plan_approve_execute_and_undo_real_file_chain` | `safe_chain` | 通过 |
| `test_approved_cross_workspace_plan_cannot_execute` | `safe_chain` | 通过 |
| `test_file_changed_after_approval_is_rejected_before_execution` | `safe_chain` | 通过 |
| `test_restart_replays_duplicate_without_move_and_can_undo` | `safe_chain` | 通过 |
| `test_batch_partial_failure_compensates_only_completed_move` | `safe_chain` | 通过 |
| `test_unapproved_status_leaves_complete_disk_snapshot_unchanged[WAITING_APPROVAL]` | `unapproved_disk_immutability` | 通过 |
| `test_unapproved_status_leaves_complete_disk_snapshot_unchanged[REJECTED]` | `unapproved_disk_immutability` | 通过 |
| `test_missing_approval_leaves_complete_disk_snapshot_unchanged` | `unapproved_disk_immutability` | 通过 |
| `test_mismatched_approved_plan_leaves_disk_unchanged` | `unapproved_disk_immutability` | 通过 |

## 结论与限制

- 4 个未经审批场景的完整磁盘快照前后相同，因此检测到的磁盘变更为 0。
- 已验证查询、规划、审批、执行、undo、越界拒绝、文件变化拒绝、幂等、重启和部分失败补偿。
- 重启场景使用全新 Engine/Session，不等同于完整服务进程重启。
- 部分失败使用确定性的目标冲突注入，不代表生产并发压力测试。

## 可复现命令

在仓库根目录、且目标路径尚不存在时运行：

```powershell
.\.venv\Scripts\python.exe -m backend.app.safe_organization_evaluation_cli --output-dir backend\backups\e26-safe-organization-v1 --report-path backend\evaluation\safe_organization_milestone.md
```
