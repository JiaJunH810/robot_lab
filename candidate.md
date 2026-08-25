# RobotLab 候选参数名单

更新时间：2026-08-26（Asia/Ulaanbaatar）
任务：Cyborg HP Flat locomotion
候选原则：P0/P1/P2 是独立替代方案，不在同一次实验中叠加。

## 合格基线与当前 HEAD

合格行走父基线：`2026-08-25_11-27-55/model_3000.pt`

```text
phase_ref_joint_pos.weight = 1.0
stand_still.weight = -0.1
stand_still joints = J_hip_.*、J_knee_.*
base_acc.weight = 0.2
action_smoothness.weight = -0.003
```

该基线有完整的静态站立、受力恢复和动态行走测评，行走能力与 phase=1.0 的关系也已有直接证据。

当前 Git HEAD：`e2d088b`（candidate-03，包含 P0 配置）。本次工作区已切回合格基准，并应用 P2：

```text
phase_ref_joint_pos.weight = 1.0
stand_still.weight = -0.1
stand_still joints = 髋/膝
joint_deviation_hip_l1.weight: -1.0 → -1.25
```

本次 P2 相对 `2026-08-25_11-27-55` 只改变 `joint_deviation_hip_l1`；P0/P1 不在本配置中叠加。

## P0：完成当前 stand_still 中间强度验证（独立候选）

- 候选类型：奖励候选
- 父版本：`2026-08-25_11-27-55`
- 旧值：`weight=-0.1`，关节为髋/膝
- 新值：`weight=-0.2`，关节为髋/膝/踝
- 证据等级：`supported / conditional`
- 选择状态：独立候选；不与当前 P2 叠加

主目标：降低站立时踝、膝和髋的 `joint_q_std`、`joint_dq_rms`，改善身体高度波动和左右对称性。

护栏：必须无跌倒且完整结束；行走时保留足端抬脚、触地相位、低同侧重复接触率和有效命令切换；同时检查 walking Roll/Pitch/Yaw、脚底滑动、落地速度、`vx/vy/wz` 误差和双向受力释放。

可能副作用：过度约束站立姿态会让行走膝踝速度、落地冲击、Yaw/漂移和命令切换稳定性变差。若接近 `-0.3` 的行走退化，应回退到 `-0.1`，而不是继续加大惩罚。

## P1：降低 phase 参考约束到中间值（未应用）

- 候选类型：奖励候选
- 父版本：`2026-08-25_11-27-55`
- 旧值：`phase_ref_joint_pos.weight=1.0`
- 新值：`phase_ref_joint_pos.weight=0.8`
- 证据等级：`supported`（方向有直接历史证据，具体 0.8 为中间值推断）
- 选择状态：独立候选；不与当前 P2 叠加

主目标：在保留 phase=1.0 行走能力的同时，回收身体高度、Roll/Yaw、对称性和站立关节位置抖动的部分代价。

护栏：必须保留明确的摆腿和触地时序；检查 touchdown 相位/频率、腾空比例、落地竖直速度、transition peak tilt、walking `vx/vy/wz` 误差、横向/竖直漂移以及受力恢复。

可能副作用：phase 参考变弱可能导致抬脚不足、步态周期漂移、重复触地或速度跟踪变差。不能直接降到 `0.2`，因为历史上虽然静态和部分行走指标更好，但其行走稳定性与 phase=1.0 的权衡很明显。

## P2：加强髋部 Yaw/Roll 回中（已应用）

- 候选类型：奖励候选
- 父版本：`2026-08-25_11-27-55`
- 旧值：`joint_deviation_hip_l1.weight=-1.0`
- 新值：`joint_deviation_hip_l1.weight=-1.25`
- 关节集合：保持 `J_hip_.*_yaw`、`J_hip_.*_roll`
- 证据等级：`hypothesis`
- 选择状态：已应用到当前工作区；基于 `2026-08-25_11-27-55`

主目标：减少髋部 Yaw/Roll 偏离默认姿态，改善静态和行走时的左右对称、Yaw 偏置、Yaw 角速度和根部漂移。它是一个比改变命令分布更局部的机制测试。

护栏：必须无跌倒且保持横向/转向速度跟踪；检查 `walking_vy/wz` 误差、髋部 Yaw/Roll 的 `joint_q_std`/`joint_dq_rms`、触地相位、脚底滑动、命令切换姿态和双向受力恢复。不得把 hip pitch、膝或踝加入该项。

可能副作用：该项没有 stand_still 的零速度门控，会同时影响行走；过强会限制横向迈步、转身和必要的髋部侧向补偿，导致速度跟踪或步态相位退化。因此先试约 25% 的增强 `-1.0 → -1.25`，不建议直接到 `-1.5`。

## 暂不建议继续调的变量

- `base_acc`：`0.2` 已优于历史 `0.1/0.3` 的整体站立结果。
- `action_smoothness`：`-0.0025` 到 `-0.00325` 没有单调收益；更平滑经常换来 Yaw 漂移、姿态或对称性退化。
- `vel_mismatch_exp`：实现只直接约束竖直速度和 Roll/Pitch 角速度，不直接解决水平漂移或 Yaw 漂移。
- 继续扩大 `stand_still` 或同时改多个奖励：会失去行走退化的归因能力。

应用状态：P2 已应用到当前工作区；P0/P1 不在本配置中叠加；未修改 `training_log.md`、未启动训练。
