# 专题 · DeepSeek DSA：稀疏注意力从架构到 kernel 的完整落地

> **学习目标**——读完能回答：
> ① MLA 已经压住了 KV cache，长上下文的瓶颈还剩哪一个？② 稀疏注意力两大家族（推理时挂选择器 vs 原生可训练）各付什么代价，DSA 站在哪、凭什么便宜？③ Lightning Indexer 的打分公式每一项是什么，哪些设计是"为 kernel 让路"？④ 两阶段续训各训什么、 indexer 为什么从计算图 detach？⑤ DSA 靠什么证据说"不掉点"，kernel 与部署生态长什么样？⑥ 十个算子各自的数学、设计理由与 kernel 形态是什么（§5–§6）？⑦ DSA 在分布式训练布局中长在哪、通信账怎么算（§8）？⑧ Megatron 与 cudnn-frontend 如何把 DSA 算子合同化（§10）？
>
> **前置知识**：注意力的 softmax 公式；G01 的 KL 散度（§0 一行带回）。**§0 数学加油站**补复杂度计数与 MLA 最小背景。
> **预计时间**：全篇 120–140 分钟，可分两读：§0–§7（原理与训练）、§8–§16（系统视角）。**归处**：G26（Attention 算子全谱）稀疏注意力小节的先行素材——工作急用，插班整理；关联 G07（长上下文）、G21/G22（并行与显存）、G28（FP8）、G29/G32（推理与服务）。
> 本文事实与数字除标注"推算"外，均核对自文末一手资料（整理于 2026-10）。

---

## 0. 数学加油站：四块过路费

**① 注意力一分钟回顾。** 每个 query 向量 $q$ 对每个前文 token 的 key 向量 $k_s$ 打分，softmax 归一化后当权重，对 value 向量 $v_s$ 加权平均：

$$\mathrm{Attn}(q) = \sum_s \mathrm{softmax}_s\!\left(\frac{q \cdot k_s}{\sqrt{d}}\right) v_s$$

**KV cache**：推理时前文各 token 的 k/v 算过一次就存下来，之后每生成一个新 token 只算新 token 的 q/k/v——不重复计算，但要付存储。

**② 复杂度怎么数。** $L$ 个 query，每个对 $L$ 个 key 打分，共 $L \times L$ 次打分——这就是注意力的 $O(L^2)$。代数字找感觉：$L = 128\mathrm{K}$ 时每个 query 要对十几万个 key 打分；若只对 $k = 2048$ 个打分，计算量比值 $2048/131072 = 1/64$。$O(L^2) \to O(Lk)$ 说的事就一件：**不是每个 query 都必须把全本前文看一遍**。

**③ KL 散度一行带回**（G01 §4）：$D_{KL}(p \,\|\, q) \geq 0$，衡量"按错误的信念 $q$ 生活比按真相 $p$ 多付的代价"——本课拿它当训练 indexer 的对齐目标，同一块料二次使用。

**④ MLA 最小背景**（完整推导在 G06）。DeepSeek 从 V2 起的注意力压缩方案：每个 token 不再为每个头存 k/v，只存一个 512 维**latent**（潜变量——把多头 k/v 的信息压进一个共享向量）加 64 维带位置部分，用时再升维还原；KV cache 由此缩到 GQA 的几分之一。**MQA 模式**（multi-query attention，多查一）：升维后的 k/v 由该 token 的**所有 query 头共享**。这两块背景足够本课使用。

## 1. 问题：MLA 拿走了显存，计算量还在

MLA 解决的是**存不下**（KV cache），但没解决**看不过来**：每个新 query 仍要对全部前文 key 打一次分，注意力计算量随 $O(L^2)$ 涨。上下文到 128K 的时代，这成了主瓶颈——prefill（把整段输入过一遍模型）算不过来，decode（逐 token 生成）中注意力算子随 token 位置越来越慢。DeepSeek-V3.2-Exp 技术报告的 Figure 3 给了实测：H800 集群按 $2/卡时 折价，稠密 MLA 的**每百万 token decode 成本随 token 位置陡增**（128K 处比开头高一个量级），prefill 成本同样随位置攀升。

一句话：**长上下文的墙从"存储墙"移到了"计算墙"，DSA 拆的是第二面**。

## 2. 路线图：稀疏注意力的两大家族

让每个 query 只看一部分前文——"稀疏注意力"。按"训练动不动"分两派：

| 家族 | 代表 | 做法 | 代价 |
|---|---|---|---|
| 推理时稀疏（训练不动） | Quest（2024）一系 | 模型照旧，推理时挂一个"检索器"挑重要的前文看 | 零训练成本；但模型训练时学的是"看全部"，推理时被迫"看局部"，**长上下文掉点** |
| 原生稀疏可训练 | NSA、MoBA（均 2025-02） | 训练时注意力就是稀疏的，模型自己学会"该看谁" | 质量保住；但要**从头训练**——NSA 为此设计了压缩/选中/滑窗三分支加门控的重结构 |

**DSA（DeepSeek Sparse Attention，2025-09 随 DeepSeek-V3.2-Exp 发布）的站位：第二家的质量，接近第一家的迁移成本。** 它不在预训练阶段介入，而是给训好的稠密大模型（V3.1-Terminus，671B、128K 上下文就绪）"做手术"：加一个超轻量索引器，**两阶段续训共约 946B token** 完成迁移（续训 = 从已有 checkpoint 接着训，不是从零开始）。而且它要过的不只预训练关——后训练（RL）也全程稀疏且不掉点，这是 NSA 一系从未验证过的关口。

## 3. 架构：Lightning Indexer + 细粒度 token 选择

### 3.1 打分公式

DSA 的原型是两件套。第一件，**闪电索引器（Lightning Indexer）**——一个极小的打分网络，给 query token $h_t$ 与每个前文 token $h_s$ 算"相关度分" $I_{t,s}$：

$$I_{t,s} \;=\; \sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\big(q^I_{t,j} \cdot k^I_s\big) \tag{1}$$

逐项拆开（维度取自官方发布配置 `index_n_heads=64, index_head_dim=128`）：

- $q^I_{t,j} \in \mathbb{R}^{128}$：索引器 query，共 $H^I = 64$ 个头——注意与主注意力的 128 个头相比已是半价，且**复用 MLA 的低秩 query 路径**（$h_t \to$ 降秩 1536 维 $\to$ RMSNorm $\to$ 索引器自己的投影，见官方推理 demo `inference/model.py`）；
- $k^I_s \in \mathbb{R}^{128}$：索引器 key——**所有头共享同一个 128 维向量**（$h_s$ 过一个小线性层，demo 里接 LayerNorm；论文未规定 norm 种类，以官方 demo 为准）；
- $w^I_{t,j} \in \mathbb{R}$：64 个标量权重（另一个小投影 $h_t \to 64$ 维，fp32 计算）——把各头分数加权求和，头可正可负地互相抵消。

**为什么是 ReLU？** 论文明说：**为了吞吐**——ReLU 对 kernel 友好、易与 FP8 融合。这是"算法为 kernel 让路"的第一次亮相，§4 还有一串。

第二件，**细粒度 token 选择**：对每个 $h_t$，按 $I_{t,s}$ 取 top-k（$k = 2048$，且不超过已有上下文长度），只对这些 KV 条目 $\{c_s\}$ 做注意力：

$$u_t \;=\; \mathrm{Attn}\Big(h_t,\; \big\{\, c_s \,\big|\, I_{t,s} \in \text{Top-k}(I_{t,:}) \,\big\}\Big) \tag{2}$$

（$c_s$ 是 MLA 的 latent KV 条目。）**"细粒度"是关键词**：选择按 token 逐个做、每个 query 独立——不是 NSA 那种 64-token 块为单位整选整弃。同一个 query 可以从上下文任意位置零散挑出 2048 个 token。

### 3.2 落在 MLA 的 MQA 模式上

DSA 要续训，就必须与已训好的 MLA 结构兼容，因此**实例化在 MLA 的 MQA 模式之上**：每个 latent 条目 $c_s$ 被该 token 的所有 query 头共享。论文引用 NSA（Yuan et al., ACL 2025）说明动机：**kernel 层面 KV 条目必须能被多个 query 复用**——稀疏 attention 的 kernel 要按索引 gather（按索引取数）KV，若 KV 不共享，gather 的访存代价会吃掉稀疏收益。这是 NSA 与 DSA 两条独立工作殊途同归的结论，也是"算法-硬件协同"的实证。附录里还交代了 V3.1-Terminus 的模式分工：训练与 prefill 用 MHA 模式、decode 用 MQA 模式；DSA 统一落在 MQA。

### 3.3 一笔诚实账：indexer 自己仍是 $O(L^2)$

每个 query 要对所有前文 token 打分——索引器没有绕过平方复杂度。它便宜是因为**常数因子小**：头数 64（主注意力 128）、维度 128、纯打分无 value 参与、全程 FP8、ReLU 融合友好。渐近上真正降维的是主注意力那一步：$O(L^2) \to O(Lk)$。（工程视角：这也是 DSA 的天花板之一——上下文再涨一个量级时，打分本身会重新变成瓶颈，见开放问题。）

## 4. 工程视角：一份"为 kernel 让路"的设计清单

这一节是本专题最值得反复品的部分：DSA 的每个算法选择背后都有一条硬件理由。

1. **ReLU 而非别的激活**——吞吐（§3.1 已述）；
2. **FP8 打分全链路**（FP8 = 8 位浮点，G28 专题）：官方 demo 的路径是——q/k 先过 **Hadamard 旋转**（一类正交旋转，把离群值摊平，FP8 量化前的标准缓解），再按 128 维块做 FP8 量化，key 的 FP8 值与 scale 存入专用 cache；打分走 FP8 GEMM kernel（开源于 DeepGEMM PR #200），量化 scale 全部折叠到 query 侧；
3. **部分非交错 RoPE**：索引器 128 维中只有前 64 维加 RoPE，且是**非交错**布局——与 MLA 主注意力的交错布局不同。这里出过真实的坑：2025-11-17 HF 模型卡发布更新，指出照抄 MLA RoPE 写法的第三方实现全是错的，indexer 必须用非交错版。**"同一个模型里两套 RoPE 布局"是复现 DSA 的第一号陷阱**；
4. **短序列 prefill 用 masked MHA 模拟 DSA**：短上下文时物理稀疏（按索引 gather）的调度开销不划算，于是稠密算注意力、再叠加 top-k 掩码（选中的位置 0、未选中的 $-\infty$，softmax 后等价于只看选中者）——同一个数学，两套 kernel 策略按场景切换；
5. **跨 DP rank 的索引一致性**：demo 里 top-k 索引从 rank 0 广播并 assert 全 rank 相等——数据并行各副本必须选出一模一样的 token，否则注意力结果分叉（G21/G29 的并行一致性议题在这里落了一次地）。

**两笔推算账**（按发布配置与 demo 存储方式估算，非官方数字）：

- **indexer 参数**：每层 query 投影 1536×8192 + key 投影 7168×128 + 权重投影 7168×64 ≈ 14M，61 层共约 **0.85B，占总参数 685B 的 0.12%**——"Lightning"之名的量化注脚；
- **indexer 缓存**：每 token 每层 128 维 FP8 + fp32 scale ≈ 132 B，相对 MLA 的 FP8 latent cache（576 B/token/层）约 **+23%**——SGLang 为此在显存池里专门开了 key & key_scale cache。

**kernel 与部署生态**（DSA 拆成多个开源件，本身就是"算法-硬件协同"的完整样本）：

- **FlashMLA**：稀疏 prefill kernel `flash_mla_sparse_fwd`（无效索引用 -1 或越界值表示，支持逐 token 可变选择数）；decode 走 `flash_mla_with_kvcache(..., indices=...)`，索引按页编码。**稀疏 decode 强制 FP8/FP4 KV cache（不支持 bf16）**——带宽算的。发布时 prefill 峰值 640 TFLOPS、decode 峰值 410 TFLOPS（H800 世代）；
- **DeepSelect**：高性能 top-k kernel（DSA 的选择步），比 `torch.topk` 快 2–20×（top-k 是访存受限算子），支持 CUDA 与昇腾。仓库注明 DSA 用于 V3.2、V4、V4.1，**V4 已把 topk 收紧到 512**；
- **TileLang kernel**：官方发布时同步开源的 TileLang 版 DSA kernel，现居 tile-ai/tilelang 的 examples（`deepseek_v32`、`deepseek_nsa`、`dsa_sparse_finetune`、`dsa_hisa`，配套 `hadamard_transform`、`topk`），官方定位是研究原型路线（⚠️ 勿与 deepseek-ai/TileKernels 混淆——那是通用 kernel 库，不含 DSA 专用件）；
- **推理引擎**：vLLM 与 SGLang 均在发布当日支持。SGLang 披露的细节很 representative：同一后端内**混合页大小**——indexer 的 key cache 用 page size 64，token 级稀疏前向用 page size 1（G30 的分页管理在此撞上现实）；部署 `--tp 8 --dp 8 --enable-dp-attention`；
- 一个减法细节：**发布的 V3.2-Exp 权重不含 MTP 模块**（配置无 nextn 字段；V3.1 的 MTP 被摘掉），SGLang 把 MTP 解码列为后续工作——DSA 与投机解码的合体留给了后人。

## 5. 算子分解：从公式到 kernel

§4 讲"为什么这样设计"，本节讲"执行时每一步跑什么"。推理时每层多出一条**索引支路**，与 MLA 主注意力支路并行，最后把"选哪些 token"的索引喂给稀疏注意力主算子：

```
每个 chunk / 每个解码步，每层：
┌─ A 前置组 ──── indexer 三个投影 + norm + 部分RoPE + Hadamard + FP8量化
├─ B 打分组 ──── fp8_index：FP8 GEMM 打分 + ReLU + 加权求和 → I(t,s)
├─ C 选择组 ──── topk(2048) → 索引（短序列改走 mask 构造）
├─ D 主算子 ──── 稀疏注意力：prefill / decode 两个 kernel 形态
└─ E 缓存/调度 ── KV 追加、indexer cache 追加、页表、索引编码
```

| 组 | 算子 | 生产 kernel | 类型 | 受限于 |
|---|---|---|---|---|
| A | 投影/norm/RoPE/Hadamard+FP8 量化 | 融合 kernel（FlashMLA 2026 提供 norm-RoPE-投影-cast 融合，B200 1460 TFLOPS） | GEMM+elementwise | 带宽 |
| B | 打分（ReLU+加权和） | `fp8_index`（DeepGEMM PR #200） | FP8 GEMM | 计算（常数极小） |
| C | token 级 top-2048 | DeepSelect（2–20× vs torch.topk） | 选择 | **访存** |
| D1 | 稀疏 prefill 注意力 | `flash_mla_sparse_fwd`（FlashMLA）/ TileLang / FA3-sparse | gather+online softmax | 计算（640 TFLOPS 级） |
| D2 | 稀疏 decode 注意力 | `flash_mla_with_kvcache(indices=)` + `get_mla_metadata` | gather+MQA+FP8 反量化 | **反量化/带宽** |
| E | cache 追加/页表/一致性 | 引擎层（SGLang/vLLM），混合页 64/1 | 调度 | 带宽+同步 |

一句话：**DSA = 三个瘦投影 + 一个 FP8 打分 GEMM + 一个访存优化的 top-k + 两个稀疏注意力主 kernel + 一套混合页缓存管理**。真正的新算子只有 B/C/D 三组；A 和 E 是把新算子缝进现有 MLA/引擎栈的胶水，而工程难点恰恰集中在胶水上。两个主 kernel 值得展开：

**Prefill：`flash_mla_sparse_fwd`**。签名 `(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)`，THD 变长风格（无 batch 维，变长请求拼接）；无效索引不用 -1 而用 `topk_length` 逐 token 控制有效数（尾部槽位是"毒化"值、永不读——gather 内循环免分支）；返回 `(out, max_logits, lse)` 供分块在线 softmax 拼接；索引升序保证 gather 沿页局部性访问。

**Decode：FP8 稀疏 kernel**（FlashMLA 仓库有专门 deep-dive）。decode 一步只算一两个 query，算子从计算受限翻转为**访存受限**——所以强制 FP8/FP4 KV：每 token 每层 656 B = 512 维 FP8（1×128 tile 量化 + 4 个 fp32 scale）+ 64 维 RoPE 部分保 bf16（位置敏感不量化）。kernel 内反量化成了新瓶颈：FP8→BF16 无直转指令，要走 FP8→half→fp32→bf16→乘 scale（约 50 cycle/token），比 MMA（约 34 cycle）还贵。解法是"crossover"：集群 2 个 CTA 各管 128 头中的一半，各反量化一半 K/V，再用 Hopper 分布式共享内存（`st.async` + cluster 事务栅栏）互换——反量化开销对半砍，410 TFLOPS（旧版 250）。与稠密 decode 的盈亏平衡点约 3000 token，128K 时选择率 1/64。`get_mla_metadata` 把 tile 调度计划离线成元数据，每 batch 只算一次。

**训练侧是另一套 kernel**（缺反向、缺变长 batch 效率，推理件不能直接用）：TileLang/Triton 谱系（见 §4 的 TileLang kernel 条目），加上复用 NSA 时代演化出的稀疏 fwd/bwd。**稠密热身阶段更是零稀疏 kernel**——主注意力稠密，只加 indexer 前向与 KL reduction，系统上是"免费"的增量实验。官方教学 demo（HF `inference/model.py`）与生产的分界：demo 用普通 matmul + scatter 掩码 + 稠密 einsum 讲原理，生产全是融合 kernel——读懂 demo 后要去生产件里找对应物。

## 6. 算子原理逐讲

把 §5 的地图拆到单个算子，每个讲三层：**数学算什么 → 为什么这么设计 → kernel 是什么形态**。十个算子背后是一个贯穿的设计公理——**选择是离散的、不可导的、但必须是确定性的**；围绕这个离散点：低精度做选择（FP8 打分、radix top-k），高精度做学习（重算+fp32 loss），LSE 做组合协议，KL 做绕过不可导的训练信号，页编码与广播做分布式的离散一致性。

**① 投影组（q/k/w + norm + 部分 RoPE）**。三个瘦线性层造打分原料（§3.1 已拆维度）。为什么 key 全头共享：打分 GEMM 里 64 个 query 头打同一个 key，key 只读一次——"KV 必须可复用"原则在 indexer 上提前兑现。为什么部分 RoPE：位置对"相关性判断"只是软特征，留一半无位置维让内容匹配不受旋转干扰。为什么 $w$ 逐 query 动态：让每个 query 自己决定"信哪个头的判断"，可正可负、头间可抵消。kernel：瘦 GEMM 单独跑会被启动与访存开销吃光，全部折叠进融合 kernel。

**② Hadamard 旋转 + FP8 块量化**。FP8 只有 4 位尾数、±448 范围，离群维会撑爆整块 scale。Hadamard 是正交变换、保范数（$H H^\top = nI$），把离群能量摊平到所有维——块内 max 变小、量化误差均匀（G28 的"先旋转再量化"配方）。块大小 128 一石三鸟：= head_dim（一向量一 scale）、= GEMM tile 宽度、= 向量访存宽度的整数倍。

**③ 打分算子（IndexerForward / `fp8_index`）**。式 (1) 一步算完，外加 **ratio 因果掩码**（bottom-right 对齐：因果性相对序列末端定义，chunk 化后 query 行号须换算到全局再施掩码——Megatron 的 CP bug 全发生在这一条边界规则上，见 §10）。ReLU 为吞吐：MMA 累加后一次比较清零，可与 epilogue 融合；非负性给分数语义——每头贡献一份非负"证据"，加权求和后仍是证据加总。**敢用 FP8 的理论底气：分数只进 top-k，只有序没有值**——排序对量化噪声天然鲁棒（分数差明显的对手不会被相对误差翻转）。

**④ Top-k 选择**。每个 query 行选最大 k 个，**不可导——这是整个训练设计的出发点**：梯度穿不过选择，才有 KL 对齐这条替代训练路（⑦⑧⑨）。三条硬合同：bitwise 可复现、平局取最小 id、升序输出——离散输出没有浮点容差，下游 gather 与跨 rank 一致性全靠"同样的分数永远选出同样的集合"。输出用 compact+`topk_length`（取代 -1 填充）——gather 内循环无分支化。kernel 是 radix 选择（8 bit 一趟 256 桶直方图，两趟定界，$O(N)$ 无全排序），访存受限所以 DeepSelect 靠带宽优化拿 2–20×。融合方向（cudnn 的 `indexer_topk`）加 force_first/force_last——强制保留首/尾 token，是 attention sink（开头 token 天然吸走注意力质量）与局部窗两个经验现象的参数化。

**⑤ 稀疏注意力前向（kernel 细节 §5 已展开）**。三个原理级要点：**在线 softmax**——不物化 L×L 概率矩阵，流式过选中 KV 块、维护运行最大值与运行和、随时重标定累加器（FlashAttention 根基；稀疏版只是"顺序流过全部 KV"换成"按索引流过选中 KV"）；**LSE 是组合协议不是副产品**——跨 chunk/跨 CP rank 的部分结果靠 LSE 加权合并成全局 softmax，训练的目标重算（⑦）直接复用它；**MQA latent 形态**——KV 每 token 一个 576 维 latent（K=V 同体）服务全部 128 头，gather 一次服务全员。`attn_sink` 是可学习的逐头加性偏置（对应④的 force_first，且参与反向）。

**⑥ 稀疏注意力反向**。梯度只在选中集合上流动——dkv 在未选中 token 上天然为零，**稀疏不仅省前向也省反向**。概率重算是命门：前向没存概率矩阵，反向重算 QK 得分，此时 lse 兑现价值——概率用 $\exp(s - \mathrm{lse})$ 恢复，无需重扫全行求 max/和。$d\mathrm{sink}$ 出现在输出里：sink 是被训练的参数。kernel 按 MLA latent 维特化（SM90 上 head_dim 限定 {512, 576}——SMEM/register tile 形状 = 架构形状）。

**⑦ 目标重算（SparseAttnScoreRecompute → $p_{t,S_t}$）**。式 (4) 的目标分布：主注意力概率**跨头求和**，再在选中集合上 L1 归一：

$$p[t,i] = \frac{\sum_{h} \exp\big(s_h[t,i]\cdot\mathrm{scale} - \mathrm{lse}_h[t]\big)}{\sum_{i' \in S_t}\sum_{h} \exp\big(s_h[t,i']\cdot\mathrm{scale} - \mathrm{lse}_h[t]\big)}, \qquad i \in S_t$$

为什么跨头求和：选择全头共享，"哪里重要"必须由全体头的集体意见定义。为什么用 $\exp(s-\mathrm{lse})$：数值稳定的概率恢复（lse 已含 max 平移，不溢出）。分母只对选中集合求和：目标与预测定义在同一支撑集上，KL 才良定义。wrapper 内做 $-\log_2(e)\cdot\mathrm{lse}$ 预处理是 kernel 惯技——GPU 上 `exp2` 快，换底系数折进预处理、主循环只留 exp2。

**⑧ 预测重算（SparseIndexerScoreRecompute → $\mathrm{softmax}(I_{t,S_t})$）**。③算过的分数为什么不复用？③走 FP8/bf16 的"选择精度"，loss 反传要精度可控、与参考实现逐位对齐的重算——**双精度原则：低精度选、高精度学**。ReLU 留在 softmax 内部有个微妙后果：预测是"非负证据的 softmax 竞争"，不是标准能量模型——这正是 DSA 打分器的概率语义。稠密热身的 Dense 变体（式 3）= 去掉选中集合限制的 ⑦⑧，故要额外吃 L1 分母张量（`attn_l1norm`/`index_l1norm`）。

**⑨ Indexer 反向（IndexerBackward）**。KL 对 softmax 输入的梯度是经典形态 $\hat p - p$（交叉熵型）——KL 对齐在数学上是"让 indexer 的 softmax 模仿主注意力的集体分布"的**分布匹配回归，不是排序学习**（不要求 top-k 顺序一致，只要求分布质量对齐）。再往参数走三条路：$w$（线性）、$q$（过 **ReLU 门**：梯度只在 $q\cdot k>0$ 处存活）、$k$（对称，且只在选中 token 上非零）。**detach 语义在此落实**：此算子的梯度只来自 $\mathcal{L}_I$，语言建模 loss 永远走不到这里。kernel 三段流水：score-grad（算 $\hat p-p$ 与 ReLU 门控）→ 三个梯度 GEMM（d_q/d_w/d_k）融进一个 warp-specialized kernel（共享同一 SMEM tile）→ 纯 torch 的 dtype cast。

**⑩ 索引编码、页管理与跨 rank 一致性**。索引是一维全局 token id，物理寻址两级翻译（页号 × 页大小 + 页内偏移）。**混合页大小**是访问模式的妥协：indexer key cache 按页 64（批量追加、顺序友好），token 级稀疏前向要精确到单 token（页 1）——同一后端两套页表并存（SGLang 的现实解）。**跨 rank 一致性**：凡注意力被切分的部署（dp-attention、TP 切头），各 rank 必须选出逐位相同的索引，否则连 softmax 分母都不同——解法是"单点计算 + 广播 + assert"（HF demo 的 rank0 广播），④的 bitwise 确定性让这个 assert 可判定。训练布局（无 TP/CP）下此算子退化为空操作；DSA 一进 Megatron 的 CP 世界它就复活成头号难题（§10）。

## 7. 训练：两阶段续训 + 全程稀疏的后训练

出发点：**上下文已扩到 128K 的 V3.1-Terminus**；训练数据分布与 Terminus 的 128K 长上下文扩展数据完全一致——变量隔离，效率与质量的变化才能干净地归因给 DSA（这套"控制变量"的报告纪律本身值得学）。

### 7.1 阶段一：Dense Warm-up——只训 indexer（1000 步 / 2.1B token）

主注意力保持**稠密**，冻结模型全部参数，**只训 indexer**。目标是让索引器打分对齐主注意力的注意力分布。目标分布的构造分两步：把主注意力的分数**跨所有注意力头求和**，再沿序列维 **L1 归一化**，得到 $p_{t,:}$；然后最小化

$$\mathcal{L}_I = \sum_t D_{KL}\Big( p_{t,:} \,\Big\|\, \mathrm{Softmax}(I_{t,:}) \Big) \tag{3}$$

学习率 1e-3，1000 步，每步 16 条 128K 序列，共 **2.1B token**——indexer 就学会了"该看哪里"。注意用 KL 不用 MSE 的理由：索引器打分最终要过 softmax 参与竞争，对齐的是分布形状而非裸分数值（G01 的 KL 直觉在此兑现）。

### 7.2 阶段二：Sparse Training——全参数 + 真稀疏（15000 步 / 943.7B token）

token 选择机制生效（$k=2048$），**所有参数一起训**，让模型适应稀疏模式。indexer 继续与主注意力对齐，但 KL 只在**被选中的集合** $S_t$ 上算：

$$\mathcal{L}_I = \sum_t D_{KL}\Big( p_{t,S_t} \,\Big\|\, \mathrm{Softmax}(I_{t,S_t}) \Big) \tag{4}$$

（没选中的 token 反正不会被看，对齐它们没有意义——损失函数与稀疏机制自洽。）关键工程决策：**indexer 的输入从计算图 detach**（切断梯度）——indexer 只由 $\mathcal{L}_I$ 优化，主模型只由语言建模 loss 优化，两条梯度路互不干扰：既防止 indexer 为讨好 LM loss 而"作弊"，也避免两股梯度打架搅乱训练。学习率 7.3e-6，15000 步，每步 480 条 128K 序列，共 **943.7B token**。

两阶段合计约 946B token——对比 NSA 的"从头训练"，这是 DSA 迁移成本的核心账本。

### 7.3 后训练：RL 也在稀疏注意力上做

后训练管线、算法、数据与 V3.1-Terminus **完全相同**（继续隔离变量），全程稀疏：

1. **专家蒸馏**：为数学、竞赛编程、通用逻辑推理、agentic coding、agentic search 等领域各训一个专用模型（均从 V3.2 稀疏基座微调 + 大规模 RL），再用专家产出最终 checkpoint 的领域数据；
2. **混合 RL**：仍用 GRPO（G14 精读），但把推理、agent、人类对齐**合并为单阶段**训练——规避多阶段 RL 的灾难性遗忘；奖励为规则化结果奖励 + 长度惩罚 + 语言一致性奖励，通用任务用每个 prompt 自带评分细则的生成式奖励模型。

**最有说服力的证据是 RL 训练曲线**：V3.2-Exp 与稠密的 Terminus 在 BrowseComp（准确率约 0.32→0.40）和 SWE Verified（约 0.64→0.68）上的曲线**几乎重合、同步上升**——稀疏注意力不仅没伤预训练能力，连 RL 阶段的训练动力学都与稠密版一致。

## 8. 分布式训练视角：DSA 长在哪张并行网格上

V3.2-Exp 的训练配置对齐 Terminus，继承 V3 的基础设施（HAI-LLM 框架，2048 张 H800）：

| 维度 | 取值 | 对 DSA 的含义 |
|---|---|---|
| PP（流水线） | 16 路，DualPipe 调度 | DSA 整体落在流水段的 **attention 块**内部 |
| EP（专家并行） | 64 路，跨 8 节点，all-to-all（dispatch/combine） | 与 DSA 无交集（只碰 MoE 块） |
| DP | ZeRO-1（优化器状态切分） | indexer 新参数照常复制 + 状态切分 |
| **TP** | **无** | 关键：注意力不切头，每 rank 拥有全部 128 头 + 完整 indexer，本地算完整个注意力 |
| CP/SP | 无（每 rank 扛整条 128K 序列） | DSA 不与 Ring Attention 交互，选择全在单卡内完成 |

**核心结论：DSA 对分布式布局零侵入**——§5 的五组算子全是本地计算，**新增集合通信 = 0、新增 P2P = 0**。不切 KV、不切头、选择在单卡完成：并行策略、通信库、DualPipe 调度一行不改，换的只是 attention 块内部调用的 kernel。这是"给训好的稠密 checkpoint 续训"在系统层成立的原因（对比：用 CP 解决长上下文会引入 P2P 链与 KV 重排，侵入整个编排）。两个布局敏感点：KL 目标 $p$ 的构造要跨**全部 128 个注意力头**求和——无 TP 布局下纯本地（若切头就得 all-reduce），训练公式与布局互相成全；跨 rank 索引一致性只在"注意力被切分"的形态出现（§6 之⑩），训练布局下不存在。

**几笔账（推算/分析，非论文口径）**：indexer 0.85B 参数的 AdamW 状态 ≈ 16 B/参数 → 全网新增约 13.6 GB，摊到 DP 组每卡几十 MB，可忽略；**激活显存基本无变化**——FA 类 kernel 本就不物化 $L^2$ 的 softmax 激活，稀疏化省的是计算时间与能耗（128K 下选择率 1/64），不是激活内存，别把"复杂度降了"脑补成"训练显存降了"；DualPipe 的 attention 段变便宜后与 all-to-all 段的重叠更从容，瓶颈进一步移向 MoE 段——与 DeepSeek 后续把优化火力集中在 MoE/通信上的路线吻合。

**训练与推理是两套 kernel**（§5 末段），RL 阶段 DSA 同时活在两个引擎里：rollout 侧 vLLM/SGLang（SGLang 实测 `--tp 8 --dp 8 --enable-dp-attention`，推理 kernel 全家桶），训练侧 HAI-LLM（稀疏 fwd/bwd），周期性 weight sync；**两侧选择逻辑完全相同，没有"训练稠密/推理稀疏"的 train-inference mismatch**——这是 RL 曲线能与稠密版重合的系统前提，也是相对 Quest 一系（训练不动、推理后挂选择器）的本质优势在 RL 场景的兑现。

## 9. 评测与成本：能力基本不掉，账单砍半

**能力**（V3.1-Terminus → V3.2-Exp，摘自技术报告 Table 1）：

| 能力 | 指标 | Terminus | V3.2-Exp |
|---|---|---|---|
| 通用 | MMLU-Pro | 85.0 | 85.0 |
| 通用 | GPQA-Diamond / HLE | 80.7 / 21.7 | 79.9 / 19.8 |
| 搜索 agent | BrowseComp（中/英） | 38.5 / 45.0 | **40.1 / 47.9** |
| 代码 | Codeforces Div1 rating | 2046 | **2121** |
| 代码 agent | SWE Verified / Terminal-bench | 68.4 / 36.7 | 67.8 / 37.7 |
| 数学 | AIME 2025 / HMMT 2025 | 88.4 / 86.1 | **89.3** / 83.6 |

总体**无实质性退化**；长上下文 agent 类（BrowseComp、Terminal-bench、Codeforces）反而略升。GPQA/HLE/HMMT 的下降，论文归因为 V3.2-Exp **生成的推理 token 更少**（RL 后长度分布不同），换用生成量相当的中间 checkpoint 后差距闭合。2025-12 的正式版论文（arXiv 2512.02556）补上外部验证：ChatbotArena Elo 持平、AA-LCR 比 Terminus 高 4 分、超长文的 Fiction.liveBench 获胜。

**成本**：技术报告用"每百万 token 成本 vs token 位置"曲线代替单点加速比——DSA 的 decode 成本曲线随位置大幅变平（稠密版在 128K 处陡增一个量级）。对外直接可见的结果：发布当日 **API 价格整体下调过半**（官方公告）。算子层的复杂度优化，一步传导到了服务经济学（G32 的 $/token 成本模型在现实中的第一手样本）。

## 10. NVIDIA 栈视角：Megatron-Core 与 cudnn-frontend（2026）

DSA 进入 NVIDIA 生态后有一套独立实现，三层栈：

```
Megatron-Core（模型层）        DSAttention 模块 · apply_dsa_kernel_fusion · --dsa-kernel-backend
        │ 调用
cudnn-frontend（算子合同层）    DSA.* 八个算子（CuTe-DSL Python 类，实验性）
        │ 落到
kernel 层                      前向主算子 = FlashMLA（C++，外部）
                               indexer/topk/反向/loss重算 = cuDNN CuTe-DSL/CUTLASS kernel
```

**反直觉的事实：稀疏注意力前向主算子不是 cuDNN 的，仍是 FlashMLA**（官方文档明说 production forward 是外部件）。cuDNN 补的是训练侧增量——打分、top-k、注意力反向、KL loss 全部算子化。两套栈的分工对照：DeepSeek 自家"训练 TileLang、推理 CUDA"；NVIDIA 栈"前向借 FlashMLA、训练侧 cuDNN"。

**cudnn 的 DSA 八算子**（`from cudnn import DSA`——训练计算图的标准分解，编号对应 §6）：

| 算子 | 数学 | 关键签名/约束 |
|---|---|---|
| IndexerForward | $\sum_h \mathrm{ReLU}(Q_h K_h^\top) W_h$ + ratio 因果掩码（bottom-right） | SM100+、head_dim=128、qhead_per_kv_head∈{32,64}；文档标注"非融合"；输出 fp32 |
| IndexerTopK | radix top-k、逐行长度 | topk ≤ 2048、SM100 |
| （前向主算子） | 稀疏注意力本体 | 外部 FlashMLA |
| SparseAttentionBackward | 梯度只在选中集合流动（⑥） | **kv 单张量（K=V 同体——MQA 合同化）**；THD 变长；SM90 需 head_dim∈{512,576}= MLA latent 维 |
| SparseAttnScoreRecompute | 式 (4) 目标 $p_{t,S_t}$（⑦） | 用 lse 恢复概率，$-\log_2 e\cdot\mathrm{lse}$ 预处理 |
| SparseIndexerScoreRecompute | 式 (4) 预测 softmax(I)（⑧） | "KL 训练等价"写进 API 验收合同 |
| IndexerBackward / DenseIndexerBackward | $\hat p - p$ 型梯度 + 三参数 GEMM（⑨）；Dense 版服务稠密热身（式 3） | 三个梯度 GEMM 融一个 warp-specialized kernel；H ≥ 64 |

合同级细节最能体现工程哲学：输入 bf16/fp16、**无 FP8**（DeepSeek 自家是 FP8——精度换简化的不同取舍）；bitwise 确定性、平局取小 id、compact+topk_length（"尾部毒化值永不读"）；KV 的别名 latent 直接写进张量形状。融合方向 `indexer_topk`（cudnn-frontend #793→#828→#832）把"打分后掩码 + 精确 top-k"合成一个算子，参数 activation / ratio / score_pool / force_first / force_last / cu_seqlens / q_causal_offsets——**NVIDIA 把 DSA 抽象成稀疏注意力家族的通用合同**（DeepSeek-V4 CSA、Qwen QSA、GLM-5.2/5.3、MSA 是同一算子在不同参数下的实例）。同仓库的 NSA 命名空间是另一套：块级四件套（Compression/TopK/Selection/SlidingWindow，block_size∈{16,32,64}，块分由压缩分支的 LSE 传递）——**块级+三分支 vs token 级+单路，两套 API 并排放着，架构差异一目了然**。

**Megatron-Core 侧**（PR #5099，2026-07 合入，关闭 CP 需求 issue #4878）：模块在 `megatron/core/transformer/experimental_attention_variant/`（DSAttention + `dsa_cudnn_kernels.py`：融合包装器 + **custom autograd**——前向调 FlashMLA 与 cuDNN indexer/topk，反向调 cuDNN 的 SparseAttentionBackward、IndexerBackward 与两个重算算子）；配置 `apply_dsa_kernel_fusion`（带运行时依赖/SM 检查）与 `--dsa-kernel-backend cudnn|tilelang|none`；布局以 packed THD 为第一公民。**CP 是真正难点**：索引是全局 token id，跨 rank 的 KV gather/globalize 与 indexer key 同步是一等难题；cuDNN 表达不了的掩码走手写 fallback（`_compute_indexer_scores_chunk_with_global_rows`），BSHD+CP 明确未支持。**成熟度的诚实账**（issue #7105，2026-09）：SM100 上 packed-THD+CP>1 有真数值错误（索引与 bottom-right 参考错位、全融合路径 cosine 0.365，曾被 flaky 标记在 CI 里掩盖）；但真实训练口径（GLM-5.2 SFT，EP32/CP4、seq 8192、packed）与参考实现 loss 差 ≤0.7%。版本坐标：Megatron-Core 0.19、TE 2.17.1、cudnn-frontend（libcudnn 9.21.1）。

**三个观察**（连课程）：**训练 loss 被算子化**——论文两行 KL 在 cuDNN 里是四个带确定性合同的专用算子，"算子-硬件协同"不止发生在注意力本体（G26）；**CP 是"DSA 零通信"神话的精确修正**——自家训练无 TP/CP 所以零通信，进通用框架的 CP 世界立刻付出整一个 PR 的代价还留了 bug（G21 通信量分析样本）；**前向 kernel 生态已固化**（FlashMLA 各家复用），竞争增量在训练侧反向与 loss kernel——这正是 cuDNN 押注的位置。

## 11. 变迁史（论文时间线）

| 时间 | 事件 | 解决了什么 |
|---|---|---|
| 2019–20 | Sparse Transformers（Child et al.）、Longformer | **固定模式**稀疏（局部窗+散点列）：先证明"不必看全部"也能训 |
| 2024 | Quest（Tang et al.）一系 | **query-aware 检索**式稀疏：训练不动、推理时挑——零迁移成本但长上下文掉点，路线定型 |
| 2025-02 | NSA（DeepSeek，ACL 2025）、MoBA（月之暗面） | **原生稀疏可训练**：块级选择（NSA 为 64-token 块）+ 硬件对齐 kernel，从头训练验证质量不掉 |
| 2025-09 | DSA 随 DeepSeek-V3.2-Exp | **token 级细粒度 + 极轻 FP8 indexer + 稠密 checkpoint 两阶段续训 + RL 全程验证**；API 降价过半 |
| 2025-12 | DeepSeek-V3.2 正式版（arXiv 2512.02556） | DSA 全链路定型，Elo 持平等外部验证补齐 |
| 2026 | V4 / V4.1（据 DeepSelect 仓库） | DSA 延续为家族标准，topk 收紧到 512——选择预算仍在演化 |
| 2026 | Megatron-Core PR #5099 / cudnn-frontend DSA API | DSA 进入通用训练框架生态：算子合同化（打分/topk/反向/KL loss 全算子化）与家族化（CSA/QSA/GLM 同一合同） |

谱系一句话：**NSA 证明"原生稀疏可训练"，DSA 解决"怎么把现成稠密大模型便宜地迁过去、且全管线不掉点"**。两者同源（论文互引），是同一问题的两步走。

## 12. 动手作业（L2，本周完成）

1. **手算**：某 query 对 4 个前文 token，两个注意力头的注意力概率分别为 $(0.4, 0.3, 0.2, 0.1)$ 与 $(0.1, 0.2, 0.3, 0.4)$。按式 (3) 构造目标分布 $p_{t,:}$；若 indexer 分数 $I_{t,:} = (2.0,\, 1.0,\, 0.5,\, -1.0)$，计算 $\mathrm{Softmax}(I_{t,:})$ 与 $D_{KL}(p \,\|\, \mathrm{Softmax}(I))$（自然对数）；
2. **读码**：打开 HF 仓库 `deepseek-ai/DeepSeek-V3.2-Exp` 的 `inference/model.py`，找到 indexer 的三个投影、LayerNorm、部分 RoPE、top-k 那一行；再按 §4 的推算方法自己数一遍 indexer 参数量；
3. **读码**：打开 FlashMLA 仓库 README，看 `flash_mla_sparse_fwd` 与 `flash_mla_with_kvcache` 的索引格式（无效索引怎么表示？索引怎么编码页号与页内偏移？），并回答：为什么稀疏 decode 强制 FP8/FP4 KV cache？
4. **（可选）实验**：用 SGLang 或 vLLM 部署 V3.2-Exp，找一篇超长文档，实测不同输入长度下的 prefill 时延，画出你自己的"成本 vs 位置"曲线雏形；
5. **读码（系统侧）**：打开 cudnn-frontend 的 DSA API 文档，抄下 `SparseAttentionBackward` 的张量签名，回答：为什么 kv 是单张量？head_dim 为什么限定 {512, 576}？再到 Megatron-Core 的 `experimental_attention_variant/dsa_cudnn_kernels.py` 里找 custom autograd，指出前向（FlashMLA）与反向（cuDNN）的分界线。

## 13. 开放问题（L3 素材）

- **indexer 自身仍 $O(L^2)$**：上下文向百万级走时打分会重新成为瓶颈——indexer 侧的稀疏化/层级化怎么做？（V4 把 topk 收到 512 是选择侧的演化，打分侧呢？）
- **k=2048 是全局常数**：不随任务与上下文长度自适应——选择数能否按需伸缩？代价函数是什么？
- **FP8 打分的量化误差 vs top-k 召回**：Hadamard 旋转是目前的缓解，但"分数差一点→选错 token→输出差很多"的敏感性无人系统测过（连 G28 的量化误差分析）；
- **稀疏注意力与 RL 长度分布的相互作用**：HLE/HMMT 的掉点归因到"推理 token 更少"，机制未完全解释——是稀疏改变了模型的"思考长度"偏好吗？
- **DSA × MTP/投机解码**：V3.2-Exp 发布时未合入，兼容性待验证；
- **第三方生态重写 DSA 的精度取舍**：cudnn 用 bf16 打分（无 FP8）、DeepSeek 自家用 FP8——"FP8 打分的召回敏感性"（上文已列）在生态重写场景下放大，谁对谁错需要系统性实验；
- **CP × DSA 的正确性合同**：Megatron SM100 packed+CP 的索引错位 bug（#7105）说明"全局索引 × 切分 KV"的边界规则（bottom-right 对齐、全局行号换算）还没有像 LSE 那样的组合协议级保障——值得设计一个。

## 14. 问答卡

1. **"DSA 到底省在哪？"** → 主注意力计算量 $O(L^2) \to O(Lk)$（$k{=}2048$，128K 时约 1/64）；indexer 虽仍 $O(L^2)$，但头少、维低、无 value、FP8——常数因子小一个量级。
2. **"和 NSA 什么区别？"** → 粒度（token 级 vs 64-token 块级）、索引器（极轻单路 vs 三分支门控）、迁移（稠密 checkpoint 续训 946B token vs 从头训练）、验证（RL 曲线与稠密重合 vs 未做 RL）。
3. **"质量掉了吗？"** → benchmark 总体持平，长上下文 agent 类略升；GPQA/HLE/HMMT 因生成推理 token 更少略降，可比 checkpoint 复平；正式版 Elo 持平。
4. **"为什么建在 MLA 的 MQA 模式上？"** → kernel 效率：稀疏 gather 要求 KV 条目被多个 query 头共享，否则访存吃掉收益（NSA 同结论）。
5. **"训练为什么要两个阶段、indexer 为什么要 detach？"** → 先稠密热身（只训 indexer，2.1B token 学"看哪里"）再全参稀疏适应（943.7B token）；detach 让 indexer 只吃 KL 对齐 loss、主模型只吃 LM loss，两条梯度路互不干扰。
6. **"DSA 在 Megatron/cuDNN 里怎么用？"** → Megatron-Core 的 DSAttention 模块 + `--dsa-kernel-backend cudnn`；前向仍走 FlashMLA，cuDNN 补 indexer/topk/反向/KL loss 算子；THD packed + CP 路线仍在收敛。
7. **"为什么说 DSA 对分布式布局零侵入？"** → 五组算子全是本地计算、零新增集合通信；但前提是"无 TP/CP"的 DeepSeek 自家布局——进 CP 世界后索引一致性立刻成为头号难题。

## 15. 自测

1. 写出 indexer 打分公式 (1)，说明每一项的维度与来源；为什么用 ReLU？
2. indexer "轻"体现在哪四处？为什么说"indexer 仍是 $O(L^2)$"与"DSA 降到 $O(Lk)$"不矛盾？
3. 两阶段续训各训什么参数、各用多少 token、学习率各多少？式 (3) 的目标分布 $p$ 是怎么两步构造的？
4. 式 (4) 为什么只在选中的集合上算 KL？detach 切断的是什么风险？
5. DSA 与 NSA 的三个本质区别是什么？各自适合什么场景？
6. 为什么短序列 prefill 用 masked MHA 模拟而不是物理稀疏？ indexer 的 RoPE 与 MLA 的 RoPE 有什么坑？
7. 为什么 top-k 的输出必须 bitwise 确定、平局取最小 id？（提示：离散输出没有浮点容差；跨 rank 的 assert 靠它可判定）
8. lse 分别在⑤前向输出、⑥反向、⑦目标重算里各起什么作用？（组合协议 / 概率恢复 / 数值稳定的目标构造）

## 16. 归处与衔接

本专题是 **G26（Attention 算子全谱）** 稀疏注意力小节的核心素材：§5–§6 的算子地图与十算子原理链，到 G26 时与 FA1/2/3、MLA kernel 两形态、Ring Attention 放在同一张"算子-硬件协同"地图上重读——重点看"算法选择被 kernel 现实倒逼"的证据链（ReLU、FP8、MQA、masked MHA 模拟）。§8 分布式视角连 G21（并行策略与通信量手推——"零新增通信"为什么值钱、CP 为什么难）与 G22（ZeRO 与优化器状态账）；§10 NVIDIA 栈是"算子进入生态"的当代样本（训练 loss 算子化、确定性合同、家族化）。更早的连接：G07（长上下文：MLA 管显存、DSA 管计算量的分工）、G14（GRPO 与 RL 兼容性证据）、G28（FP8 与 Hadamard）、G29/G32（成本曲线到 API 定价的传导）。

## 资料出处

- 技术报告：DeepSeek-AI《DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention》（2025-09-29，[GitHub 仓库 PDF](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)，6 页，公式/训练量/评测表均出自本文）；
- 正式版论文：《DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models》（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)，DSA 章节与 Exp 报告一致，补充 Elo 等外部验证）；
- 官方实现：[HF 模型卡与推理 demo](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)（`inference/model.py`、`config_671B_v3.2.json`、2025-11-17 RoPE 修复说明）；
- kernel：[FlashMLA](https://github.com/deepseek-ai/FlashMLA)（稀疏 prefill/decode，性能数字见 README；FP8 稀疏 decode 的 656 B 格式与 crossover 细节出自仓库 docs 的 deep-dive）、[DeepSelect](https://github.com/deepseek-ai/DeepSelect)（top-k kernel，V4 topk=512 出处）、DeepGEMM PR #200（FP8 indexer 打分）；训练侧 TileLang kernel 见 [tile-ai/tilelang examples](https://github.com/tile-ai/tilelang/tree/main/examples)（deepseek_v32 / deepseek_nsa / dsa_sparse_finetune / dsa_hisa）；
- NVIDIA 栈：[cuDNN Frontend DSA API](https://archive.docs.nvidia.com/deeplearning/cudnn/v1.24.0/fe-oss-apis/dsa.html)（八算子签名与约束）、[NSA API](https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/nsa.html)（块级四件套对照）、[cudnn-frontend #793/#828/#832](https://github.com/NVIDIA/cudnn-frontend/issues/832)（generic sparse_attention 与融合 indexer_topk 设计线、家族形状矩阵）、[Megatron-LM PR #5099](https://github.com/NVIDIA/Megatron-LM/pull/5099)（DSAttention 接线）、[#4878](https://github.com/NVIDIA/Megatron-LM/issues/4878)/[#7105](https://github.com/nvidia/megatron-lm/issues/7105)（CP 需求与 SM100 成熟度）；
- 部署：[SGLang Day-0 支持博客](https://www.lmsys.org/blog/2025-09-29-deepseek-V32)（混合页大小、后端矩阵）；vLLM day-0 支持；
- 发布公告：[DeepSeek API 文档 2025-09-29](https://api-docs.deepseek.com/news/news250929)（API 降价）。
