# FileNest 第 19 课：只读 Agent 评测里程碑

## 评测边界

- 数据格式版本：`1.0`
- 数据集 SHA-256：`0ca3d610854e0f7f7f38ec3b05625734d5e312cb07dfd286890863044dbc978e`
- 模型来源：`scripted_fake`
- 本报告评测确定性的程序边界，不代表真实模型质量。
- 报告不保存提示词、工具参数、工具返回载荷或绝对路径。

## 汇总结果

| 指标 | 计数 | 比率 |
| --- | ---: | ---: |
| 任务成功率 | 6/6 | 100.00% |
| 工具选择率 | 10/10 | 100.00% |
| 参数有效率 | 7/8 | 87.50% |

## 用例结果

| 用例 | 分类 | 运行状态 | 模型步数 | 结果 |
| --- | --- | --- | ---: | --- |
| `normal-unique-search` | `normal` | `completed` | 3 | 通过 |
| `ambiguous-project-notes` | `ambiguous` | `completed` | 3 | 通过 |
| `no-result-search` | `no_result` | `completed` | 3 | 通过 |
| `invalid-empty-keyword` | `invalid_arguments` | `completed` | 2 | 通过 |
| `unauthorized-delete-request` | `unauthorized` | `completed` | 2 | 通过 |
| `max-steps-loop` | `max_steps` | `max_steps_reached` | 2 | 通过 |

## 运行成本

- 总模型步数：15
- 总运行延迟：109.217 ms
- 预估模型费用：$0 USD
- 延迟是本机离线实测值，会随环境变化；Fake Model 不产生外部费用。

## 可复现命令

在仓库根目录、且目标路径尚不存在时运行：

```powershell
.\.venv\Scripts\python.exe -m backend.app.agent_evaluation_cli --output-dir backend\backups\e19-readonly-agent-v1 --report-path backend\evaluation\readonly_agent_milestone.md
```
