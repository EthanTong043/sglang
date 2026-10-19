# SGLang DFS 主 KV 有界写回执行器

2026-10-20。本地候选，尚未构建/部署或证明服务性能收益。

分支 `fix/dfs-bounded-backup-20261020`，基线
`b728b7b7a5c7e091001cd09863065afe264d8611`。本次只改
`python/sglang/srt/managers/cache_controller.py`，新增专用 CPU 测试；
Hybrid 的 sidecar、ACK、collective 和 host allocator 源码未改。

## 实现与配置

在现有 HiCache storage extra config 中加入：

```json
{"backup_io_workers": 2}
```

默认 `1` 保留原串行路径，不创建执行池；显式支持 `2`、`4`，先验 `2`。
并发开关只允许 Mooncake DFS-only、direct IO、page_first_direct 和原生
v1 primary write 入口。其他 backend/layout/模式及非整数配置拒绝启动。

operation 仍由唯一协调线程 FIFO 执行。Hybrid sidecar 仍先按原边界同步写完；
之后同一 operation 内最多 B 个主 KV 子批交给持久 ThreadPoolExecutor。
每批最多 128 pages，不一次提交整个 operation。legacy draft 留在协调者，
保持已确认主 KV 前缀顺序和既有 best-effort 语义。

worker 只调用 `batch_set_v1`，不碰 radix、host allocator、ACK 或 collective。
每个批次有独立 prefix_keys 快照。返回向量须长度准确且元素都是 bool；
逐页结果保存在 operation 上，但 completed_tokens 仍只确认连续的完整成功批。
后批成功、前批失败时，后批真实成功单列，不能谎称未写或计入连续 prefix。

失败设置停止提交信号；stop 后不再提交新批；已经提交的 future 全部收敛后
才返回/抛出。异常沿用固定运行时的 no-ACK 行为，不提前解除 host 保护。
detach 的原 10 秒 join 若未完成会明确失败，并保留 backend/执行器供重试；
成功停止协调线程后才 shutdown 池、关闭 backend。reset 同样先 join 再换池。

## 字节与生命周期边界

主 KV 同时提交 payload 上界：

`B × 128 × page_size × primary_host_pool.size_per_token`

真实样本每页 774,144 B 时，B2 为 198,180,864 B（189 MiB）/rank；
B4 为 396,361,728 B（378 MiB）/rank。该值包含 primary host pool 中实际
打包的层，未用旧3.375MiB总层逻辑页替代 CP slice。

这是在途主 KV payload 上界，不是总 L2 容量、native staging 上界或整个
operation 的保护字节上界。INDEXER/其他 sidecar 仍按原提交边界串行执行，
不会与这组 primary worker 重叠；legacy draft 可能与其余 primary worker
重叠，但协调者已先取走一个完成的 future，不额外增加主 KV pending 数。

host 保护从原入队延续到 operation 的所有任务退出及原 ACK 消费；ACK 解锁
并不等于删除仍可命中的 L2 缓存。这个版本没有新增 operation 队列限额或
改变驱逐规则，应继续观测 queued/active/awaiting_ack、不可回收占用和排空。

## 计量与共享状态审查

- 原 `_DfsWriteMetrics` 聚合、repair state 均有 mutex；pool metric context
  使用 thread-local。keys/pointer lists/ReplicateConfig 都是逐调用创建。
- `operation.backup_io_stats` 记录提交/峰值 pending/峰值实际执行、worker
  queue wait、累计执行时间、协调者等待、成功页及未进入连续 prefix 的成功页。
- `operation.backup_batch_results` 保存起始页、大小、逐页结果或异常类型。
- 复用现有 trace 白名单：`backup_batch_begin` 的 elapsed_s 为执行队列等待、
  queue_depth 为当时 active；`backup_batch_end` 为执行耗时；
  `backup_operation_io_end` 为 coordinator_wait、peak active、成功页和
  primary payload 上限；`backup_operation_pending_peak` 为 peak pending 与
  未确认成功页。trace 默认关闭且有事件上限，只是诊断采样；连续性能判断
  仍用现有 app/native Prometheus 计量，payload 上限不是实写字节。
- 旧 `backup_pgs`/`backup_bandwidth` 列表仍采用 append + copy/clear，属于
  继承的近似采样，不保证跨线程无丢样，不能据此声称实写 GPFS 带宽。
- 固定源码本来就 import `prefetch_trace_runtime`，该辅助文件来自历史
  runtime overlay，未跟踪在本分支。打包必须保留已验证版本；历史辅助函数
  使用锁/有界队列且诊断异常不传播。本次未引入新的 trace 依赖或修改它。

## 本地验证

在本工作树根运行：

```bash
python3 test/registered/unit/mem_cache/test_bounded_dfs_backup.py
python3 test/registered/unit/mem_cache/test_mooncake_dfs_batch_placement.py
python3 test/registered/unit/mem_cache/test_mooncake_dfs_self_heal.py
git diff --check
```

结果：有界写回 15/15、writer/ACK/计量 24/24、DFS repair 59/59 通过。
专用并发测试再重复 25 轮，375/375 通过。一次补测曾暴露测试观察点早于
协调者提交计数更新的竞争；改为执行屏障检查 active、结束后检查提交总数，
生产断言与成功口径未变。
专用测试使用 production AST 和独立假后端，不依赖 CUDA/native；以 Event
验证真实 B2/B4 重叠、pending 上界、乱序失败、异常/no-ACK、提交失败、draft
异常、stop/detach/reset、sidecar 先行及 B1 兼容。未放宽既有测试断言。

这不能证明 native client/RPC/staging/pybind GIL 的同 client 并发安全或性能，
也不能代替真实 CP4 collective、host transfer、跨 Prefill 恢复验证。

## 集成下一步

主线程先完成同 native client B1/B2 的写入、readiness、发布/回收和读写混合
验证，再把该实现与确定 native 版本打入完整候选镜像。新生命周期先2P1D，
检查真实 simultaneous native IO、字节/hash、sidecar、ACK 保护与排空，再
进行同条件 B1/B2 服务对照。B4 不优可以保留 B2；不能仅以 Python 线程重叠
声明吞吐提高。回退使用新生命周期的 `backup_io_workers=1`，不热改当前 run。

本子任务没有远端操作、镜像构建、commit/push、MR 合并或共享状态修改。
