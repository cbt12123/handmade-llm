"""
03 · 位置编码：注意力天生"看不出顺序"，必须把位置信息塞进去
==============================================================
注意力是集合运算（对输入顺序置换等变），所以必须显式注入位置信息。
三种主流方案：
  ① 正弦位置编码（原版 Transformer）  sin/cos 不同频率
  ② 可学习绝对位置（BERT / GPT-2）    就是一张 nn.Embedding(max_len, d)
  ③ 旋转位置编码 RoPE（Llama / 现代 LLM）：对 Q/K 做旋转，使内积只依赖相对位置

本脚本手写三者，并演示 RoPE 为什么"天然支持相对位置"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, has_matplotlib, save_fig, section, set_seed, subsection

set_seed(2)


# ----------------------------------------------------------------------------------
def sinusoidal_pe(max_len, d_model):
    """PE[pos, 2i]   = sin(pos / 10000^(2i/d))
       PE[pos, 2i+1] = cos(pos / 10000^(2i/d))"""
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


section("1) 正弦位置编码")
MAX_LEN, D = 50, 64
pe = sinusoidal_pe(MAX_LEN, D)
print(f"  PE 形状 {pe.shape}")
print(ascii_heatmap(pe[:32], width=60, height=16, title="  前 32 个位置的编码（每行一个位置）："))
print("  低频维（左侧）变化慢，高频维（右侧）变化快 —— 类似二进制计数器的多分辨率表示。")

subsection("关键性质：PE(pos+k) 可以由 PE(pos) 线性表示 → 模型能学到相对位置")
k = 5
A = np.linalg.lstsq(pe[:MAX_LEN - k], pe[k:], rcond=None)[0]
recon = pe[:MAX_LEN - k] @ A
err = np.abs(recon - pe[k:]).max()
print(f"  用线性变换从 PE(pos) 预测 PE(pos+{k}) 的最大误差 = {err:.4f}")
print("  误差很小 → 相对位置关系是线性的，一层线性层就能表达。")

section("2) 可学习绝对位置编码（BERT / GPT-2 的做法）")
rng = np.random.default_rng(0)
learned = rng.normal(0, 0.02, (MAX_LEN, D))
print(f"  本质就是 nn.Embedding({MAX_LEN}, {D})，参数量 = {MAX_LEN * D}")
print("  优点：灵活；缺点：不能超过训练时的最大长度（外推能力差）")

# ----------------------------------------------------------------------------------
section("3) RoPE 旋转位置编码（现代 LLM 的事实标准）")


def rope(x, pos, theta=10000.0):
    """对向量 x（维度 d 为偶数）施加位置 pos 的旋转。

    做法：把相邻两维看成复平面上的一个点，旋转角度 pos·θ_i。
    """
    d = x.shape[-1]
    half = d // 2
    freqs = 1.0 / (theta ** (np.arange(0, half) / half))       # (d/2,)
    angles = pos * freqs                                       # (d/2,)
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


q = rng.normal(size=(D,))
k_vec = rng.normal(size=(D,))
print("  RoPE 的核心性质：<R_m·q, R_n·k> 只依赖 (m − n)")
for (m, n) in [(0, 0), (3, 7), (10, 14), (5, 13)]:
    lhs = float(rope(q, m) @ rope(k_vec, n))
    rhs = float(rope(q, 0) @ rope(k_vec, n - m))
    print(f"    m={m:>2}, n={n:>2}: <R_m q, R_n k>={lhs:+.6f}   <R_0 q, R_{{n-m}} k>={rhs:+.6f}   差={abs(lhs - rhs):.2e}")
print("  两者完全相等 → 相对位置信息被编码进了 Q·K 的内积里，不需要额外的位置向量。")

section("4) 三种方案对比")
print(f"""
{'方案':<16}{'是否可外推':<12}{'额外参数':<12}{'代表作'}
{'-' * 62}
{'正弦 PE':<16}{'可以':<12}{'0':<12}{'Transformer(2017)'}
{'可学习绝对':<16}{'不行':<12}{'L·d':<12}{'BERT / GPT-2'}
{'RoPE':<16}{'较好':<12}{'0':<12}{'Llama / Qwen / Mistral'}
{'ALiBi':<16}{'好':<12}{'0':<12}{'BLOOM / MPT（在注意力分数上减距离惩罚）'}
""")

section("5) 长上下文外推：为什么要有 NTK / YaRN / 线性插值")
print("""
  训练时最长 4k，推理想用 32k → RoPE 的频率没见过那么大的角度 → 注意力分数乱掉
  三种补救：
    线性插值 PI    : 把位置索引按比例压缩回训练范围（简单但损失分辨率）
    NTK-Aware     : 非均匀缩放高频/低频（高频少缩放，保留局部分辨率）
    YaRN          : NTK + 注意力温度补偿，Llama-2/3 长上下文扩展常用
  实践结论：想稳，优先选"原生支持长上下文训练"的模型，而不是靠外推。
""")

if has_matplotlib():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(pe[:32].T, cmap="RdBu", aspect="auto")
    ax.set_xlabel("position")
    ax.set_ylabel("dim")
    ax.set_title("Sinusoidal Positional Encoding")
    save_fig(fig, Path(__file__).parent / "outputs" / "03_positional_encoding.png")
