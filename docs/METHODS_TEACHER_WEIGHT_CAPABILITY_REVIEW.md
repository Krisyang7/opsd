# OPSD 系列方法对比：教师信息、权重与局限

> 更新日期：2026-11-02
>
> 范围：以 OPSD 及其直接变体为主；GRPO 只作为必要基线。重点回答：教师给什么、怎样传给学生、位置如何加权、全错时怎么办。

## 1. 核心方法一张表

| 方法 | 教师或监督信息 | 怎样训练学生 | 位置权重 | 主要价值 | 主要小问题 |
|---|---|---|---|---|---|
| [GRPO](https://arxiv.org/abs/2402.03300) | 没有教师；只有最终结果 reward | 同题多条 rollout 计算组相对 advantage | 同一 rollout 的 token 共享同一个 advantage | 方向由可验证结果决定，简单稳定 | 全错/全对组信号退化；无法定位错误 step；答案碰巧正确也可能奖励错误过程 |
| [OPSD](https://arxiv.org/abs/2601.18734) | 同一模型教师额外看到标准解 | 教师在学生原前缀上重评分，学生拟合全词表分布 | 通常按位置等系数聚合，但各位置实际 KL 不同 | 不需要更大教师；提供 dense token 信号 | 单一参考路线偏置；可能学习 PI 捷径或措辞；PI 教师也不会时无解 |
| [U-OPSD（Unsupervised OPSD）](https://arxiv.org/abs/2608.06296) | 多次 rollout 的多数答案及其一致轨迹 | 多数轨迹作 pseudo-solution；只蒸馏与多数答案不一致的轨迹 | 先做题目/rollout 筛选，选中轨迹内仍是 dense token loss | 不需要标准答案、verifier 或外部教师 | 多数可能稳定地错；低共识和全部一致样本被跳过；多次采样成本高；不定位具体错误 step |
| [RLSD（RLVR with Self-Distillation）](https://arxiv.org/abs/2604.03128) | verifier 决定方向；同模型教师额外看到标准答案 PI 并给出 sampled-token 概率比 | GRPO advantage 决定正负，教师只调节每个 token 的更新幅度 | 每个 token 幅度不同，但同一 rollout 的更新符号仍相同 | 避免让 PI 教师直接控制更新方向；兼顾 RLVR 稳定性和细粒度 credit | 全错组 advantage 为 0；错误轨迹中的正确步骤仍只能少受惩罚；依赖 verifier、PI 和额外前向 |
| [AOPSD（Adaptive OPSD）](https://arxiv.org/abs/2609.32667) | 根据学生进度，只给教师参考 reasoning DAG 的已达子图和下一 frontier | 高分歧 token 触发教师短续写，再由无 PI 学生评分并做 band-pass 加权 | 题目课程、PI 披露和 token 权重都随学生能力变化 | 同时调整“教师看到什么”和“学生听多少” | DAG 构造和审核成本高；依赖单一参考路线；进度判断可能循环；probe 权重仍是启发式 |
| **当前 Dual-Gated 方法** | 结果 verifier、教师过程认证、可信 PI/sibling | 正确分支走 GRPO；失败分支计划使用 PI-conditioned teacher | 正确分支已定义；失败分支尚未确定 | 先阻止错误过程获奖或污染 sibling 池 | 同模型认证可能循环自证；PI 匹配、repair loss 和 span/token 权重仍未闭环 |

### 最短区别

| 方法 | 谁决定更新方向？ | 谁决定 token 更新大小？ | 没有标准解能否训练？ | 全错组能否直接训练？ |
|---|---|---|---|---|
| GRPO | verifier reward | 不区分，整条 rollout 相同 | 否 | 否 |
| OPSD | PI 教师分布 | PI 教师分布差异 | 通常不能 | **有可靠 PI 且教师会用时可以** |
| U-OPSD | 多数一致轨迹 | 多数条件教师分布 | 可以 | 只有形成可靠共识时可以 |
| RLSD | verifier/GRPO | PI 教师概率比 | 否 | 否，基础 advantage 为 0 |
| AOPSD | PI 教师对 sampled token 的 log-prob gap | 部分 PI + continuation band-pass 权重 | 否，需要验证参考解/DAG | PI 教师能沿 DAG 提供有效指导时可以 |

---

## 2. 教师到底给学生什么？

| 方法 | 教师看到什么 | 学生看到什么 | 优势 | 风险 |
|---|---|---|---|---|
| GRPO | 无教师 | 题目 | 没有 PI 泄漏 | 只有稀疏结果信号 |
| OPSD | 题目 + 完整标准解 + 学生前缀 | 题目 + 学生前缀 | 把标准解转成学生真实状态上的 dense 信号 | 完整标准解可能诱导路线模仿和 PI 捷径 |
| U-OPSD | 题目 + 多数一致的 pseudo-solution + 不一致轨迹前缀 | 题目 + 不一致轨迹前缀 | 无人工标签也能形成教师优势 | 共识不等于正确；会放大共同错误 |
| RLSD | 题目 + 标准答案 PI + 学生前缀/已采样 token | 题目 + 学生前缀 | 教师只做“力度调节器”，不直接规定输出分布 | 概率比可能因 PI 过长、能力不足而噪声很大 |
| AOPSD | 题目 + 学生已达 DAG 子图 + 下一 frontier | 题目 + 学生前缀 | 目标是让教师只领先一个 frontier，降低过度提示 | DAG/进度判断错时会给错层级的 PI |
| 当前方法 | 计划使用外部标准流程或经过认证的 sibling | 题目 + 自己的前缀 | 可结合可靠性和路线兼容性选择 PI | “可信 sibling”仍不自动等于“适合当前轨迹的教师” |

OPSD 类方法不是把参考解直接交给学生做 SFT，而是：

~~~text
学生先生成自己的前缀
→ 教师额外读取 PI
→ 教师在同一个学生前缀上给 next-token 信号
→ 只更新无 PI 的学生
→ 测试时移除 PI
~~~

---

## 3. 每个 step/token 的权重是否相同？

| 方法 | 题目/rollout 级 | step/span 级 | token 级 | 实际判断 |
|---|---|---|---|---|
| GRPO | 组内 reward 决定 rollout advantage | 无 | 同一 rollout 使用同一 advantage | **显式同权**，不能定位关键 token |
| OPSD | 通常所有采样轨迹都蒸馏 | 无可靠 step 定位 | 通常等系数聚合 KL；实际梯度因 KL 不同而不同 | 不是“实际完全同权”，但没有显式信用分配 |
| U-OPSD | 按共识阈值选题，只选 disagreement rollout | 无 | 选中轨迹内做 dense 分布匹配 | 找到错误轨迹，不等于找到错误 step |
| RLSD | sequence advantage 仍来自 verifier | 无 | 用 \((P_T/P_S)^{\operatorname{sign}(A)}\) 调节幅度 | 有 token 权重，但不能改变 sequence-level 正负方向 |
| AOPSD | 按学生能力排题 | DAG 进度决定披露 frontier | 高分歧位置用短续写和 band-pass 权重 | 最细，但权重表示“可学习性”而非已证明的因果责任 |
| 当前建议 | 先判断题目/PI 是否可靠可学 | 只选择验证成功的 repair span | 首版 span 内等系数 | 先做二值 mask，再把连续 token 权重作为消融 |

因此，“所有 token 显式同权一定错误”仍是一个过强前提。更稳妥的首版是：

~~~text
题目级 gate
→ 验证后的 step/span 二值 mask
→ span 内先等系数
→ 连续 token 权重后续再比较 RLSD/AOPSD 等方案
~~~

---

## 4. 如果 8 条 rollout 全错怎么办？

先纠正一个前提：0/8 不等于真实成功率为 0。如果单次真实成功率为 10%，8 次全错的概率仍约为 43%。

| 方法 | 0/8 时的情况 | 能否产生信号？ | 没有解决的部分 |
|---|---|---|---|
| GRPO | 所有 reward 相同 | 否 | 冷启动和知识缺失 |
| OPSD | 仍有外部标准解作为 PI | PI 教师能理解并利用 PI 时可以 | 教师也不会、损坏前缀无法恢复时不行 |
| U-OPSD | 可能形成“多数错误答案”，也可能没有稳定共识 | 有共识时会训练，但方向未必正确 | 共同错误可能被强化；真正未知知识无法产生 |
| RLSD | GRPO advantage 为 0 | 否，token 权重乘 0 仍是 0 | 不解决 all-wrong 冷启动 |
| AOPSD | 有验证参考 DAG，可逐层披露 frontier | 教师能利用局部 DAG 时可能可以 | 同模型能力不足或 DAG 判断错误时仍失败 |
| 当前方法 | 先增加搜索，再测试 PI-teacher/repair 是否成功 | repair 能被独立验证时才训练 | PI 教师也失败时必须外部知识/工具或 mask |

### 能力诊断

| 状态 | 判断方式 | 处理 |
|---|---|---|
| 搜索不足 | 增加采样后偶尔成功 | 扩大搜索；使用经过认证的成功 rollout |
| PI 可唤醒潜在能力 | 无 PI 失败，给参考后教师稳定成功 | 使用 OPSD/AOPSD 式指导，并检查无 PI 迁移 |
| 局部可修复 | 短 hint 或局部 repair 后重跑成功 | 只训练验证成功的 repair span |
| 真正知识缺失 | 搜索、PI 教师和 repair 都失败 | 引入外部知识/工具；否则 mask |

---

## 5. 其他直接 OPSD 工作

| 方法 | 主要处理的问题 | 一句话做法 | 小问题 |
|---|---|---|---|
| [PACED](https://arxiv.org/abs/2603.11178) | 哪些题值得蒸馏 | 用学生通过率 \(p\) 设置题目权重 \(p(1-p)\) | 0/8 的 \(p\) 估计噪声大；跳过难题不等于解决难题 |
| [BRTS](https://arxiv.org/abs/2605.09725) | 选哪条 teacher rollout | 先保证正确，再选更接近学生的教师轨迹 | 文本接近不等于推理兼容；Best-of-N 成本高 |
| [Trajectory-aware OPD](https://arxiv.org/abs/2606.00305) | 单 token 差异不代表真正偏航 | 比较近未来短窗轨迹再分配指导 | continuation 成本高；短窗改善不保证最终正确 |
| [SPOT](https://arxiv.org/abs/2608.04419) | 哪个位置、哪个候选真正有用 | 对候选续写并用最终 verifier 校准目标 | 每个候选都要额外 rollout；候选全错仍无解 |
| [TA-OPD](https://arxiv.org/abs/2605.26844) | 教师信号学生能否吸收 | 只保留教师纠正仍落在学生支持内的位置 | 可学习不等于正确；可能拒绝真正的新知识 |
| [PW-OPSD](https://arxiv.org/abs/2605.21606) | 不同序列位置的监督可靠性 | 使用位置先验加权 | 廉价但不能定位具体逻辑错误 |
| [SMRC-SD](https://arxiv.org/abs/2608.05219) | 参考轨迹与学生环境状态不匹配 | 只蒸馏状态兼容的 turn | 依赖显式状态；不易直接迁移到自由文本数学 |
| [What Should a Self-Teacher See?](https://arxiv.org/abs/2609.25623) | 教师 PI 应该多详细 | 比较完整解、策略、类别、answer-only 等 PI | 没有跨任务通用的最佳 PI |
| [MCPD](https://arxiv.org/abs/2610.30880) | 单一参考视图造成目标不稳定 | 对多个 partial PI views 的教师分布做边缘化 | 平均可能模糊真正不同的解法模式 |
| [Privileged, but Biased](https://arxiv.org/abs/2608.04794) | OPSD 的参考路线和低信息 token 偏置 | 诊断 PI 如何改变教师目标 | 是诊断工作，不是完整修复算法；结论不能跨设置泛化 |
| [Critique-Conditioned OPSD](https://arxiv.org/abs/2610.13744) | scoring teacher 不会主动找错 | 先生成 critique，再让教师据此重评分 | critique 也会错；额外生成成本；证据仍较早期 |

---

## 6. 对当前方法的直接结论

| 问题 | 推荐首版 | 原因 |
|---|---|---|
| 正确 rollout 怎么训练？ | 先做答案验证和教师过程认证，再在原始 8 条组内计算 GRPO advantage | 防止答案碰巧正确的错误过程被奖励；不能把全正样本单独重组 |
| 失败 rollout 的教师给什么？ | 比较完整流程、抽象策略、经过认证且路线兼容的 sibling；教师在学生原前缀上重评分 | “正确 PI”与“适合当前轨迹的 PI”是两件事 |
| 权重怎么分？ | 题目 gate + 验证 repair span mask；span 内先等系数 | 比直接叠加 entropy、position、KL 等权重更容易验证 |
| RLSD 值得用吗？ | 作为 token-weighting 强基线 | 它能调节力度，但不能修复全错组，也不能给错误轨迹中的正确步骤正奖励 |
| AOPSD 值得借鉴吗？ | 借鉴“只给下一 frontier”和“学生可学习性” | 与当前教师 PI 设计最接近，但完整 DAG 方案成本较高 |
| U-OPSD 值得用吗？ | 作为无标签数据基线，不作为有可靠标准解时的默认方案 | 共识便宜地提供 PI，但共同错误风险无法消除 |
| 模型真的不会怎么办？ | 外部知识/工具，或 mask | 同模型权重设计不能凭空创造缺失知识 |

### 最小闭环

~~~text
学生采样 8 条 rollout
→ 最终答案验证
→ 正确答案再做教师过程认证
→ 已认证正确：保留原组计算 GRPO
→ 失败：选择可靠且路线兼容的 PI
→ PI 教师在学生前缀上提出局部 repair
→ 重跑并验证 repair 是否改善终局
→ 只训练验证成功的 span
→ PI 教师也失败：外部知识/工具或 mask
→ 在未见同技能题上做无 PI 迁移测试
~~~

---

## 7. 核心来源与证据状态

| 方法 | 论文 | 证据状态 |
|---|---|---|
| GRPO | Shao et al., 2024, [DeepSeekMath](https://arxiv.org/abs/2402.03300) | arXiv 公开版本；GRPO 来源论文 |
| OPSD | Zhao et al., 2026, [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) | arXiv 预印本；ICLR 2026 投稿版本，未核实正式录用 |
| U-OPSD | Li et al., 2026, [On-Policy Self-Distillation without Any Supervision](https://arxiv.org/abs/2608.06296) | arXiv 预印本 |
| RLSD | Yang et al., 2026, [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) | arXiv 预印本；官方标注 work in progress，当前实验范围有限 |
| AOPSD | Du et al., 2026, [Learning from a Thoughtful Teacher](https://arxiv.org/abs/2609.32667) | arXiv v1 预印本 |

注意：U-OPSD、RLSD、AOPSD 的效果数字均是作者在特定模型、数据和训练设置中的报告，不能直接当作跨模型普遍结论。

## 8. 一句话总结

> GRPO 决定“整条轨迹奖还是罚”；OPSD 让 PI 教师直接教每个 token；U-OPSD 用模型共识代替标准解；RLSD 让 reward 决定方向、教师只调力度；AOPSD 再进一步控制教师看多少 PI、学生听多少反馈。
