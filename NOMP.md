## NOMP

### 1. 信号与测量模型（输入）

假设雷达接收到的是由 $K$ 个连续空间中的复正弦波（可以理解为连续的距离或速度单元目标）组成的混合信号，并夹杂着复高斯白噪声 。系统对该信号进行了 $N$ 点均匀采样，得到观测向量 $y \in \mathbb{C}^N$：
\[
y = \sum_{l=1}^{K} g_l x(\omega_l) + z, \quad z \sim \mathcal{CN}(0, \sigma^2 \mathbb{I}_N)
\]
其中：

- $\omega_l \in [0, 2\pi)$ 是第 $l$ 个目标的连续物理频率（对应雷达空间中未对齐网格的真实距离/速度）

- $g_l \in \mathbb{C}$ 是第 $l$ 个目标的复增益（包含目标的幅度以及相位信息）

- $x(\omega)$ 是频率为 $\omega$ 的归一化导向矢量（Steering Vector / Atom），定义为：
  \[
  x(\omega) \triangleq \frac{1}{\sqrt{N}} \left[ 1, e^{j\omega}, e^{j2\omega}, \dots, e^{j(N-1)\omega} \right]^T
  \]
  因为进行了归一化，所以 $\|x(\omega)\|^2 = x(\omega)^H x(\omega) = 1$

我们的最终任务是：在不知道目标个数 $K$ 的情况下，从包含噪声的 $y$ 中精确估计出所有的 $\left\{(g_l, \omega_l): l=1,\dots,K\right\}$ 以及 $K$ 本身

### 2. 核心数学基石：单目标最大似然估计（ML）

在推导多目标循环迭代之前，我们必须先知道**如何从当前的残差信号中逼近“单一目标”的参数**

如果当前观测（或残差）里只看作有一个目标：$y \approx g x(\omega)$ ，最大似然估计（ML）的目标是最小化残差功率 $\|y - g x(\omega)\|^2$：
\[
\min_{g, \omega} \|y - g x(\omega)\|^2 = \min_{g, \omega} \left( y^H y - 2 \mathfrak{R}\{y^H g x(\omega)\} + |g|^2 \|x(\omega)\|^2 \right)
\]
这等价于最大化似然代价函数 $S(g, \omega)$：
\[
S(g, \omega) = 2 \mathfrak{R}\{y^H g x(\omega)\} - |g|^2 \|x(\omega)\|^2
\]
**步一：消除 nuisance 参数复增益 $g$**

假设频率 $\omega$ 已知，我们对 $S(g, \omega)$ 关于复增益 $g$ 求导并令其为 0，可以得到 $g$ 的闭式解（Least Squares 解）：
\[
\hat{g}(\omega) = \frac{x(\omega)^H y}{\|x(\omega)\|^2} = x(\omega)^H y \quad (\text{因为 } \|x(\omega)\|^2 = 1)
\]
> 这里的 nuisance（干扰/有害）参数是指我们真正关心的是频率 $\omega$（目标在哪里），而幅值增益 $g$ 只是一个伴生变量 。我们需要在 $\omega$ 固定的情况下，找出能让 $S(g, \omega)$ 最大的 $\hat{g}$
>
> 因为 $g$ 是复数，我们可以将其拆写为实部和虚部：$g = g_R + j g_I$
>
> 同样，把复数内积也拆开，令 $V = x(\omega)^H y$（这是一个复数标量），由于 $y^H x(\omega) = V^H = V^*$
>
> 那么，第一项可以写为：
> \[
> 2\mathfrak{R}\{y^H g x(\omega)\} = 2\mathfrak{R}\{g \cdot V^*\}
> \]
> 设 $V = V_R + j V_I$，则 $V^* = V_R - j V_I$。展开 $g \cdot V^*$：
> \[
> g \cdot V^* = (g_R + j g_I)(V_R - j V_I) = (g_R V_R + g_I V_I) + j(g_I V_R - g_R V_I)
> \]
> 取其实部：
> \[
> \mathfrak{R}\{g \cdot V^*\} = g_R V_R + g_I V_I
> \]
> 第二项 $|g|^2 \|x(\omega)\|^2 = (g_R^2 + g_I^2) \cdot \|x(\omega)\|^2$
>
> 现在，代入 $S(g, \omega)$，我们得到了一个纯实数的关于 $g_R$ 和 $g_I$ 的二元函数：
> \[
> S(g_R, g_I, \omega) = 2(g_R V_R + g_I V_I) - (g_R^2 + g_I^2)\|x(\omega)\|^2
> \]
> 为了求最大值，我们分别对 $g_R$ 和 $g_I$ 求偏导，并令偏导数为 0：
>
> 1. **对实部求导**：
>    \[
>    \frac{\partial S}{\partial g_R} = 2V_R - 2g_R\|x(\omega)\|^2 = 0 \implies \hat{g}_R = \frac{V_R}{\|x(\omega)\|^2}
>    \]
>
> 2. **对虚部求导**：
>    \[
>    \frac{\partial S}{\partial g_I} = 2V_I - 2g_I\|x(\omega)\|^2 = 0 \implies \hat{g}_I = \frac{V_I}{\|x(\omega)\|^2}
>    \]
>
> 重新组合成复数 $\hat{g} = \hat{g}_R + j \hat{g}_I$：
> \[
> \hat{g}(\omega) = \frac{V_R + j V_I}{\|x(\omega)\|^2} = \frac{V}{\|x(\omega)\|^2} = \frac{x(\omega)^H y}{\|x(\omega)\|^2}
> \]
> 因为在我们的雷达导向矢量定义中，进行了归一化，$\|x(\omega)\|^2 = 1$ ，所以：  
> \[
> \hat{g}(\omega) = x(\omega)^H y
> \]

**步二：导出广义似然比（GLRT）代价函数**

将 $\hat{g}(\omega)$ 代回 $S(g, \omega)$ 中，似然函数退化为仅与频率 $\omega$ 相关的函数，即 **GLRT 代价函数 $G_y(\omega)$**：
\[
G_y(\omega) = \frac{|x(\omega)^H y|^2}{\|x(\omega)\|^2} = |x(\omega)^H y|^2
\]
因此，寻找最优连续频率的数学本质就是：
\[
\hat{\omega} = \arg\max_{\omega \in [0, 2\pi)} G_y(\omega)
\]

> 现在我们将求得的 $\hat{g}(\omega) = x(\omega)^H y$ 和 $\|x(\omega)\|^2 = 1$ 代回原代价函数 $S(g, \omega)$ 中，看看能把 $g$ 消除成什么样 ：  
> \[
> S(\hat{g}, \omega) = 2 \mathfrak{R}\{y^H \cdot [x(\omega)^H y] \cdot x(\omega)\} - |x(\omega)^H y|^2 \cdot 1
> \]
> 我们仔细观察第一项：
>
> 由于 $[x(\omega)^H y]$ 是个标量，可以挪到前面去：
> \[
> y^H \cdot [x(\omega)^H y] \cdot x(\omega) = [x(\omega)^H y] \cdot [y^H x(\omega)]
> \]
> 因为 $y^H x(\omega) = (x(\omega)^H y)^*$，一个复数乘以它的共轭等于它的模平方：
> \[
> [x(\omega)^H y] \cdot [x(\omega)^H y]^* = |x(\omega)^H y|^2
> \]
> 既然 $|x(\omega)^H y|^2$ 本身就是一个纯实数，那么取它的实部 $\mathfrak{R}\{\cdot\}$ 还是它自己：
> \[
> 2 \mathfrak{R}\{|x(\omega)^H y|^2\} = 2 |x(\omega)^H y|^2
> \]
> 现在把它和第二项合并：
> \[
> S(\hat{g}, \omega) = 2 |x(\omega)^H y|^2 - |x(\omega)^H y|^2 = |x(\omega)^H y|^2
> \]
> 这个仅仅依赖于频率 $\omega$ 的最终函数，就是我们梦寐以求的 **GLRT 代价函数 $G_y(\omega)$** ：  
> \[
> G_y(\omega) = |x(\omega)^H y|^2
> \]

### 3. NOMP 完整流水线详细推导（多目标流程）

NOMP 采用顺序贪婪搜索。令当前迭代轮次为 $m$（初始 $m=0$），初始参数集 $\mathcal{P}_0 = \{\}$，初始残差 $y_r(\mathcal{P}_0) = y$

下面是第 $m$ 轮迭代的完整自适应推导闭环：

#### 阶段一：CFAR 终止准则检查（是否继续检测？）

在进行任何检测前，先检查当前的残差 $y_r$ 是否已经“纯粹是噪声”。论文基于恒虚警率（CFAR）推导了终止阈值 $\tau$：

1. 计算当前残差的标准 DFT 谱最大值：$\max_{\omega \in \text{DFT}} G_{y_r}(\omega) = \|\mathcal{F}y_r\|_\infty^2$

2. **阈值表达式** ：在给定的名义虚警率 $P_{fa}$ 下，利用极值渐进分布，阈值计算为：
   \[
   \tau = \sigma^2 \log(N) - \sigma^2 \log \log \left( \frac{1}{1 - P_{fa}} \right)
   \]

   >**基于极值理论的 CFAR 阈值公式**
   >
   >这个公式的数学本质是：**当信号中只剩下纯高斯白噪声时，通过 FFT（傅里叶变换）后，这 $N$ 个频点中“最大那个谱峰”的概率分布规律**
   >
   >我们把推导拆解为四个数学步骤：
   >
   >**第一步：单频点噪声能量的分布（指数分布）**
   >
   >假设经过 NOMP 的顺序剥离，所有的真实目标都已经被减掉了。此时的残差向量 $y_r$ 纯粹由复高斯白噪声组成：
   >\[
   >y_r = z \sim \mathcal{CN}(0, \sigma^2 \mathbb{I}_N)
   >\]
   >我们对 $y_r$ 进行标准傅里叶变换（FFT），得到全频域的噪声向量 $Z = \mathcal{F}y_r$。因为傅里叶变换是个正交投影矩阵，它不会改变白噪声的统计特性：
   >
   >- 每个频点 $Z_k$ 依然是独立的复高斯分布：$Z_k \sim \mathcal{CN}(0, \sigma^2)$
   >- 既然 $Z_k$ 是复高斯分布，那么它的实部和虚部都是均值为 0、方差为 $\frac{\sigma^2}{2}$ 的独立高斯随机变量
   >
   >在雷达检测中，我们关心的是**频点的功率（能量）**，即 GLRT 代价函数 $G_{y_r}(\omega_k) = |Z_k|^2$ 。 根据概率论，两个独立高斯变量的平方和服从**指数分布（Exponential Distribution）**。因此，单个频点噪声能量 $X_k = |Z_k|^2$ 的概率密度函数（PDF）和累积分布函数（CDF）为：
   >\[
   >\text{CDF: } P(X_k \le x) = 1 - e^{-\frac{x}{\sigma^2}}
   >\]
   >**第二步：$N$ 个频点中“最大值”的分布（极值理论）**
   >
   >OMP 算法每次迭代时，是在整个频域中去挑选**最强的那个谱峰**：
   >\[
   >M_N = \max_{k=0,\dots,N-1} |Z_k|^2
   >\]
   >我们想让算法停止，就必须保证这个**最大值** $M_N$ 都跨不过阈值 $\tau$。那么，$M_N$ 小于等于 $\tau$ 的概率是多少呢？
   >
   >因为这 $N$ 个频点的噪声是相互独立的，根据独立随机变量联合分布的性质，“最大值小于等于 $\tau$” 等价于 “**每一个频点都同时小于等于 $\tau$**”：
   >\[
   >P(M_N \le \tau) = P(X_0 \le \tau) \times P(X_1 \le \tau) \times \dots \times P(X_{N-1} \le \tau)
   >\\ P(M_N \le \tau) = \left( 1 - e^{-\frac{\tau}{\sigma^2}} \right)^N
   >\]
   >**第三步：引入名义虚警率 $P_{fa}$ 求解精确阈值**
   >
   >什么是虚警（False Alarm）？虚警就是指**明明场子里只剩噪声了，最大噪声峰值 $M_N$ 却还是不幸超过了阈值 $\tau$，导致算法误判其为新目标** 。因此，虚警概率 $P_{fa}$ 就是“最大值大于 $\tau$” 的概率：
   >\[
   >P_{fa} = P(M_N > \tau) = 1 - P(M_N \le \tau) = 1 - \left( 1 - e^{-\frac{\tau}{\sigma^2}} \right)^N
   >\]
   >现在，我们的目标是从这个公式里反解出阈值 $\tau$。我们一步步进行代数变形：
   >
   >1. 移项：
   >   \[
   >   \left( 1 - e^{-\frac{\tau}{\sigma^2}} \right)^N = 1 - P_{fa}
   >   \]
   >
   >2. 两边同时开 $N$ 次方：
   >   \[
   >   1 - e^{-\frac{\tau}{\sigma^2}} = (1 - P_{fa})^{\frac{1}{N}}
   >   \]
   >
   >3. 移项隔离指数项：
   >   \[
   >   e^{-\frac{\tau}{\sigma^2}} = 1 - (1 - P_{fa})^{\frac{1}{N}}
   >   \]
   >
   >4. 两边同时取自然对数 $\log$：
   >   \[
   >   -\frac{\tau}{\sigma^2} = \log \left[ 1 - (1 - P_{fa})^{\frac{1}{N}} \right]
   >   \]
   >
   >5. 两边乘以 $-\sigma^2$，得到**精确的闭式阈值表达式**：
   >   \[
   >   \tau = -\sigma^2 \log \left[ 1 - (1 - P_{fa})^{\frac{1}{N}} \right]
   >   \]
   >
   >论文中提到，这个公式已经可以直接用了。但在统计学中，为了更好地看清参数 $N$ 和虚警率对阈值的物理定性影响，我们需要利用渐进极限将其写成极值 Gumbel 分布的形式
   >
   >**第四步：渐进极值分布推导（转换为标准论文公式）**
   >
   >当雷达采样点数 $N$ 比较大时（比如 $N=256$ 或更大），$(1 - P_{fa})^{\frac{1}{N}}$ 可以利用泰勒展开（一阶近似：当 $x \to 0$ 时，$(1-x)^a \approx 1 - ax$）进行化简 ：  
   >\[
   >(1 - P_{fa})^{\frac{1}{N}} \approx 1 - \frac{1}{N} P_{fa}
   >\]
   >将这个近似代回第三步的对数项中：
   >\[
   >\tau \approx -\sigma^2 \log \left[ 1 - \left(1 - \frac{1}{N} P_{fa}\right) \right] = -\sigma^2 \log \left[ \frac{P_{fa}}{N} \right]
   >\]
   >利用对数的运算法则 $\log(\frac{A}{B}) = \log(A) - \log(B)$ 展开：
   >\[
   >\tau \approx -\sigma^2 \left( \log(P_{fa}) - \log(N) \right) = \sigma^2 \log(N) - \sigma^2 \log(P_{fa})
   >\]
   >然而，极值理论（Gumbel 分布）给出了一个更平滑、误差更小的渐进变换估计。它定义整个最大值的均值水平线在 $\sigma^2 \log(N)$ ，而在这个基准线上，根据名义虚警率需要追加的偏移量由双重对数约束：
   >\[
   >\tau = \sigma^2 \log(N) - \sigma^2 \log \log \left( \frac{1}{1 - P_{fa}} \right)
   >\]
   >
   >> $\tau = -\sigma^2 \log \left[ 1 - (1 - P_{fa})^{\frac{1}{N}} \right]$ 到 $\tau = \sigma^2 \log(N) - \sigma^2 \log \log \left( \frac{1}{1 - P_{fa}} \right)$
   >>
   >> 核心数学工具：两个重要极限和泰勒展开：
   >>
   >> 1. **等价无穷小 / 泰勒展开**：当 $x \to 0$ 时，$\log(1+x) \approx x$
   >> 2. **重要极限（对数的自然底数定义）**：当 $n \to \infty$ 时，$\left(1 + \frac{x}{n}\right)^n \to e^x$。也就是说，反过来有 $A^{1/N} = e^{\frac{1}{N}\log(A)}$
   >>
   >> 从右往左：
   >> $$
   >> \tau = -\sigma^2 \log \left[ 1 - (1 - P_{fa})^{\frac{1}{N}} \right]
   >> $$
   >> **第一步：利用指数函数改写内层幂函数**
   >>
   >> 我们将中间的 $(1 - P_{fa})^{\frac{1}{N}}$ 改写为以 $e$ 为底的指数形式。因为 $A = e^{\log(A)}$，所以：
   >> $$
   >> (1 - P_{fa})^{\frac{1}{N}} = e^{\log\left((1 - P_{fa})^{\frac{1}{N}}\right)} = e^{\frac{1}{N} \log(1 - P_{fa})}
   >> $$
   >> 把这一步代回原式，你的公式变成了：
   >> $$
   >> \tau = -\sigma^2 \log \left[ 1 - e^{\frac{1}{N} \log(1 - P_{fa})} \right]
   >> $$
   >> **第二步：利用泰勒展开化简指数项**
   >>
   >> 在雷达系统中，采样点数 $N$ 通常比较大（例如论文中设置 $N=256$） ，这意味着 $\frac{1}{N}$ 是一个趋近于 0 的**无穷小量**。因此，指数上面的那一坨 $\frac{1}{N} \log(1 - P_{fa})$ 也是一个趋近于 0 的极小值。
   >>
   >> 此时，我们可以使用麦克劳林级数（泰勒展开）：当 $x \to 0$ 时，$e^x \approx 1 + x$
   >>
   >> 令 $x = \frac{1}{N} \log(1 - P_{fa})$，那么：
   >> $$
   >> e^{\frac{1}{N} \log(1 - P_{fa})} \approx 1 + \frac{1}{N} \log(1 - P_{fa})
   >> $$
   >> 我们将这个近似关系代入公式：
   >> $$
   >> \tau \approx -\sigma^2 \log \left[ 1 - \left( 1 + \frac{1}{N} \log(1 - P_{fa}) \right) \right]
   >> $$
   >> 括号去掉，正负 1 刚好抵消，式子瞬间变得极其清爽：
   >> $$
   >> \tau \approx -\sigma^2 \log \left[ -\frac{1}{N} \log(1 - P_{fa}) \right]
   >> $$
   >> **第三步：处理负号，转换为论文的“分数形式”**
   >>
   >> 由对数里面的负号可以通过对数性质放进去：$-\log(A) = \log\left(\frac{1}{A}\right)$
   >>
   >> 我们把方括号里那个刺眼的负号处理掉：
   >> $$
   >> -\log(1 - P_{fa}) = \log\left(\frac{1}{1 - P_{fa}}\right)
   >> $$
   >> 所以公式变成了：
   >> $$
   >> \tau \approx -\sigma^2 \log \left[ \frac{1}{N} \log\left(\frac{1}{1 - P_{fa}}\right) \right]
   >> $$
   >> **第四步：拆解对数（核心变身步）**
   >>
   >> 现在，我们要利用最基本的对数运算性质：$\log(A \cdot B) = \log(A) + \log(B)$，以及 $\log\left(\frac{1}{N}\right) = -\log(N)$
   >>
   >> 我们将方括号里的 $\frac{1}{N}$ 和 $\log\left(\frac{1}{1 - P_{fa}}\right)$ 拆成两项相加：
   >> $$
   >> \tau \approx -\sigma^2 \left\{ \log\left(\frac{1}{N}\right) + \log \left[ \log\left(\frac{1}{1 - P_{fa}}\right) \right] \right\}
   >> \\ \tau \approx -\sigma^2 \left\{ -\log(N) + \log \log\left(\frac{1}{1 - P_{fa}}\right) \right\}
   >> $$
   >> 把大括号外面的 $-\sigma^2$ 分别乘进这两项：
   >>
   >> 第一项：$(-\sigma^2) \times (-\log(N)) = \sigma^2 \log(N)$
   >>
   >> 第二项：$(-\sigma^2) \times \log \log\left(\frac{1}{1 - P_{fa}}\right) = -\sigma^2 \log \log\left(\frac{1}{1 - P_{fa}}\right)$
   >>
   >> 得到右边：$\tau = \sigma^2 \log(N) - \sigma^2 \log \log \left( \frac{1}{1 - P_{fa}} \right)$
   >
   >**公式背后的物理直觉：**
   >
   >1. **$\sigma^2 \log(N)$**：代表了由 $N$ 个噪声频点竞争出来的“天生最高谱峰”的平均身高。随着雷达采样点数 $N$ 增多，噪声里冒出高头大马的概率就变大，所以阈值必须水涨船高，成 $\log(N)$ 趋势抬升
   >2. **$-\sigma^2 \log \log (\frac{1}{1 - P_{fa}})$**：代表了你的**容错安全裕量**。如果你要求虚警率 $P_{fa}$ 极低（比如 $0.001\%$），这个负负得正的项就会变得很大，迫使阈值拉高，宁可漏检也绝不虚警；反之，如果你允许一定虚警（$P_{fa}$ 设大），阈值就会下调，表现出极高的高灵敏度
   
3. **决策**：若 $\|\mathcal{F}y_r\|_\infty^2 \le \tau$，说明残差谱峰已隐没在噪声中，**算法终止**，输出当前的 $\mathcal{P}_m$ 。反之，则令 $m \leftarrow m+1$，继续下述步骤

#### 阶段二：粗检阶段 —— 离散过采样网格搜索（Identify）

由于连续空间 $\omega \in [0, 2\pi)$ 是无限维的，无法直接遍历。NOMP 先在一个离散的过采样网格 $\Omega$ 上搜寻峰值：

\[
\Omega = \left\{ k \cdot \frac{2\pi}{\gamma N}, \quad k = 0, 1, \dots, \gamma N - 1 \right\} \quad (\text{过采样因子通常取 } \gamma = 4)
\]
利用 FFT 快速计算网格上的 $G_{y_r}(\omega)$，找到最大谱峰对应的**粗检频率 $\omega_c$** 及其对应的**初始增益 $\hat{g}$**：

\[
\omega_c = \arg\max_{\omega \in \Omega} |x(\omega)^H y_r|^2
\\ \hat{g} = x(\omega_c)^H y_r
\]

#### 阶段三：单目标精细化 —— 走向连续空间（Single Refinement）

为了消除 $\omega_c$ 与真实连续频率之间的基底错配 ，以 $(\hat{g}, \omega_c)$ 为初始值，在**连续参量空间**上利用**一阶和二阶导数**进行牛顿法迭代优化（优化 $R_s$ 次，通常 $R_s = 1$）：

1. **构造当前单目标的局部似然函数**：
   \[
   S(g, \omega) = 2 \mathfrak{R}\{(y_r + \hat{g}x(\omega_c))^H g x(\omega)\} - |g|^2 \|x(\omega)\|^2
   \]
   （注：这里是将该目标提取出来，将其网格分量加回残差，从而在其局部进行导数求解 ） 

2. **计算一阶导数 $\dot{S}$ 和二阶导数 $\ddot{S}$**：
   \[
   \dot{S}(\hat{g}, \hat{\omega}) = \mathfrak{R}\left\{ (y - \hat{g}x(\hat{\omega}))^H \hat{g} \frac{dx(\hat{\omega})}{d\omega} \right\}
   \\ \ddot{S}(\hat{g}, \hat{\omega}) = \mathfrak{R}\left\{ (y - \hat{g}x(\hat{\omega}))^H \hat{g} \frac{d^2x(\hat{\omega})}{d\omega^2} \right\} - |\hat{g}|^2 \left\| [cite_start]\frac{dx(\hat{\omega})}{d\omega} \right\|^2
   \]

3. **牛顿更新步** ： 当满足局部凹性条件（$\ddot{S} < 0$）时 ，频率更新为：
   \[
   \hat{\omega}' = \hat{\omega} - \frac{\dot{S}(\hat{g}, \hat{\omega})}{\ddot{S}(\hat{g}, \hat{\omega})}
   \]
   
4. **增益同步更新**：$\hat{g}' = x(\hat{\omega}')^H y$

5. **细化接受条件（RAC）**：检查新频率下的 GLRT 值是否严格增大，即 $G_y(\hat{\omega}') > G_y(\hat{\omega})$。若满足则接受更新，否则保留原网格点参数 。最终将这个单步细化后的新目标加入集合中：$\mathcal{P}_m' = \mathcal{P}_{m-1} \cup \{(\hat{g}', \hat{\omega}')\}$

#### 阶段四：多目标循环精细化 —— 决策反馈纠错（Cyclic Refinement）

新目标的加入会打破原有所有目标参数的平衡 。因此，NOMP 会执行 $R_c$ 轮循环微调（例如 $R_c = 3$）。在一轮循环中，需要**轮流（Cycle）对集合 $\mathcal{P}_m'$ 中的每一个目标 $l$ 进行刷新**：

1. **排除第 $l$ 个目标，计算排他残差（Exclusive Residual）**：
   \[
   y_{r,\setminus l} = y - \sum_{k \neq l} g_k x(\omega_k)
   \]
   
2. **局部牛顿微调**： 将 $y_{r,\setminus l}$ 视为新的测量输入，代入**阶段三**的单目标牛顿迭代公式，对第 $l$ 个目标的频率 $\omega_l$ 和增益 $g_l$ 进行微调更新

3. 循环完所有旧目标后，重复该轮次直到达到 $R_c$ 次 。这一步完成了目标之间的非正交干扰消除

#### 阶段五：全局最小二乘投影（LS Update）

在完成了所有目标的连续空间频率微调后，此时所有的频率估计 $\{\omega_1, \omega_2, \dots, \omega_m\}$ 已经固定 。为了让残差在当前频率子空间下绝对最小 ，算法对增益矢量进行一次联合的最小二乘（LS）投影：

1. 构建当前这一轮的连续空间导向矩阵 $X \in \mathbb{C}^{N \times m}$：

   \[
   X = \left[ x(\omega_1), x(\omega_2), \dots, x(\omega_m) \right]
   \]

2. 利用广义逆（Moore-Penrose pseudo-inverse）一步更新所有目标的复增益：

   \[
   \begin{bmatrix} g_1 \\ g_2 \\ \vdots \\ g_m \end{bmatrix} = X^\dagger y = (X^H X)^{-1} X^H y
   \]

3. 此时，更新本轮最终的参数集 $\mathcal{P}_m = \{(g_l, \omega_l)\}_{l=1}^m$，并计算最新且最干净的残差：

   \[
   y_r(\mathcal{P}_m) = y - X \begin{bmatrix} g_1 \\ \vdots \\ g_m \end{bmatrix}
   \]
   
4. 

完成阶段五后，回到**阶段一**开始下一轮新目标的检测

### 4. 全流程逻辑拓扑直观图

你可以对照这个流图来设计你的一维 OMP 升级程序：

```Plaintext
===========================================================================================
                               NOMP 算法核心拓扑流图
===========================================================================================

       [ 输入：观测信号 y, 名义虚警率 P_fa, 噪声方差 σ² ]
                       │
                       ▼
             ┌───────────────────┐
             │ 初始参数化配置     │ ───►  m = 0 (迭代轮数)
             │ 初始残差 yr = y   │       P_0 = {} (估计参数集)
             └───────────────────┘
                       │
                       ▼
 ┌───► ⚡【阶段一：CFAR 终止判定】─────────────────────────────────────────────────┐
 │     计算当前残差 yr 的标准 DFT 谱峰值: M_N = ||F * yr||_∞²                       │
 │     根据 P_fa 和 σ² 动态计算红线阈值 τ                                           │
 │                                                                              │
 │     if M_N <= τ :  ───►【终止】输出当前的 P_m 及其所有参数 (收工)                  │
 │     if M_N >  τ :  ───►【继续】令 m = m + 1，进入下一轮检测                       │
 └─────────────────────┬─────────────────────────────────────────────────────────┘
                       │
                       ▼
       🎯【阶段二：粗检阶段 (Identify)】
       │   在 γ 倍（如4倍）过采样网格 Ω 上快速搜索 yr 的 GLRT 谱峰值
       │   获取最强网格点：频率 ω_c  和  初始增益 g = x(ω_c)^H * yr
       ▼
       🔍【阶段三：单目标牛顿精细化 (Single Refinement)】
       │   打破网格束缚！以 (g, ω_c) 为初值，在连续空间中局部计算一阶和二阶导数
       │   进行 Rs 次牛顿迭代更新： ω_new = ω_old - (S_dot / S_ddot)
       │   同步更新增益，若满足局部能量下降条件（RAC），则接受新参数
       │   临时加入参数集： P_m' = P_{m-1} ∪ {(g_new, ω_new)}
       ▼
       🔄【阶段四：多目标循环精细化 —— 决策反馈机制 (Cyclic Refinement)】
       │   ★ 解决近距离目标干扰的灵魂步骤 ★
       │   For loop = 1 to R_c (进行多轮整体纠错微调，通常3轮):
       │       │
       │       └─► 轮流针对当前集合 P_m' 中的每一个目标 l ：
       │               ① 计算排除掉目标 l 后的排他残差： yr,\l = y - ∑_{k≠l} g_k*x(ω_k)
       │               ② 将 yr,\l 作为输入，调用阶段三的牛顿法，重新平滑修正目标 l 的 (ω_l, g_l)
       │               ③ 用微调后的新参数更新目标 l，为下一个目标的微调提供更干净的残差
       ▼
       📐【阶段五：全局最小二乘投影 (LS Update)】
       │   锁定阶段四循环纠错后最终稳定下来的连续空间频率： {ω_1, ω_2, ..., ω_m}
       │   构建最新的连续空间流形导向矩阵： X = [x(ω_1), x(ω_2), ..., x(ω_m)]
       │   利用广义逆一次性全局刷新所有目标的复增益： [g_1, ..., g_m]^T = (X^H * X)⁻¹ * X^H * y
       │   计算本轮最终的最净残差： yr = y - X * g
       └───────────────────────┘
                               │
                               └─► 返回【阶段一】检查剩下的残差是否能继续检测
```

在这个拓扑流图中，你可以直观地看到：

- **如果只做到阶段三**：就是你目前实现的“标准离网 OMP”。当两个点间距小于 $1 \Delta_{dft}$ 时，你在阶段二找出来的第一个粗检点必然被第二个点带偏了，即使阶段三怎么用导数微调，它也只能微调到两点重叠后的“能量中心”去，第一个点彻底定错位
- **有了阶段四的循环（Cyclic）结构**：当算法在下一轮把第二个点也抓进来之后，在阶段四里，第一个点会把第二个点的影响从输入里“减掉（释放出排他残差）”。此时第一个点在没有干扰的连续空间中重新计算牛顿导数，就能精准地从“能量中心”弹回它自己真实的物理位置

### 5. 公式在系统复现中的具象含义

你在编写 MATLAB 或 PyTorch 雷达仿真代码时，这个推导过程有三个极重要的指导意义：

1. **矩阵 $X^H X$ 不是单位阵**：在离散 OMP 中，如果我们假定基底正交，$X^H X$ 接近单位阵。但是在连续超分辨率空间中，两个距离非常近的目标（$\omega_1 \approx \omega_2$），其对应的导向矢量高度相关。因此阶段五的 $(X^H X)^{-1}$ 全局 LS 投影和阶段四的循环纠错**绝对不能省**，它是算法具备**超分辨率能力**的关键

2. **导数的解析计算**：在代码实现牛顿法时，$\frac{dx(\omega)}{d\omega}$ 和 $\frac{d^2x(\omega)}{d\omega^2}$ 具有非常简单的闭式解（对 $x(\omega)$ 里的各项做虚数项乘数即可，即乘上 $j \cdot n$ 和 $-n^2$），计算极其高效，这解释了为什么它能做到 $O(R_c R_s K^2 N)$ 级别的极低计算复杂度
