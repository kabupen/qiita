
# 最尤法

点推定の推定方法として取り上げられる最尤法と、python を用いた数値計算手法についてまとめる。

## 尤度関数

$X=(X_1, X_2,..., X_n)$ の同時確率密度関数を $f_n(\bm{x},\bm{\theta})$ と表し、$\bm{\theta}$ の関数としてみたもの
$$
L(\bm{\theta}) = f_n(\bm{x},\bm{\theta}) = \prod_{i=1}^n f(x_i, \bm{\theta})
$$
を尤度関数という。最尤推定量は尤度関数を最大にする $\theta$ の値を推定値とする推定量である。
尤度関数はあるパラメータセット $\bm{\theta}$ に対する同時確率密度関数であるので、$\bm{\theta}$ に関する積分は $1$ にはならないことに留意。

## 尤度方程式

最尤法では尤度関数を最大化するパラメータを用いて推定を行う。最尤推定量 （Maximum Likelihood Estimator; MLE）$\hat{\theta}$ は
$$
\frac{\partial}{\partial\theta} L(\theta) = 0 
$$
の尤度方程式の解として与えられる。
対数尤度関数を最大化することが容易である場合が多く、
$$
\frac{\partial}{\partial\theta} \log L(\theta) = 0 
$$
も使用される。

## 「尤もらしさ」について

尤度関数はあくまでもある確率モデル（統計モデル）を想定し、それに基づいて計算されているものである。
最尤推定量 $\hat{\theta}$ とは選択した統計モデルがデータに最も適合するように計算されているものであり、その推定量が真の値かどうかには言及されていない。


### よくある例

最尤法に関する問題点を論じる際によく見かける例（ex. Bishop, p.67）で、ベルヌーイ試行を用いたものがある。

3回コインを投げて偶然3回連続で表が出たとする。表が出る確率を $p$、観測データ $\mathcal{D} = \{x_1,x_2,x_3\}$ として対数尤度関数は
$$
L(p) = \prod_{i=1}^3 p^{x_i}(1-p)^{1-x_i} \\
\log L(p) = \sum_{i=1}^3 \{ x_i\log p + (1-x_i) \log (1-p) \}
$$
以上から表が出る確率の最尤推定量は
$$
\frac{\partial}{\partial p} \log L(p) = 0  \\
\hat{p} = \frac{1}{n} \sum_{i=1}^3 x_i
$$
となるので、この場合に最尤法で表が出る確率を推定すると $\hat{p}=1$ となり、
今後は全て表が出続けると予想することになってしまう。
最尤推定量はあくまでも推定量であり、また自らが設定した統計モデルの範囲内でデータに最も適合する値を計算している。そのため、この点推定だけでなくより多角的に（$p$値、信頼区間など）評価することが必要になる。


# 数値計算による尤度方程式の計算

簡単な統計モデルを仮定した場合、尤度方程式は導関数を計算して解析的に最尤推定量 $\hat{\theta}$ を求めることができる。一般的な統計モデルではパラメータ数の増大やモデルの複雑化などから、解析的に求めることが困難である。そこで数値計算による最適化計算を行うことになる（深層学習などでも損失関数を解析的には最適化できないので様々な手法を用いて最適化している）。

以下では各種手法について Python の実装を交えながら簡単にまとめる。

## ベルヌーイ試行、二項分布

成功確率が $p$ である試行を $n$ 回繰り返し、$k$ 回成功したとき、この事象の同時確率密度関数は
$$
L(p) = {}_nC_k p^{k}(1-p)^{n-k}
$$
で表される。また対数尤度は
$$
\log L(p) = \log {}_nC_k +  k\log p + (n-k) \log (1-p)
$$
で表され、最適化計算（=極小値の算出）では負の対数尤度を用いると扱いやすい。

```python
def likelihood_bernoulli(p,n,k):
    # n : 試行回数
    # k : 成功回数
    # p : 成功確率
    return comb(n,k) * p**k * (1-p)**(n-k)

def negative_loglikelihood_bernoulli(p,n,k):
    # n : 試行回数
    # k : 成功回数
    # p : 成功確率
    return -1 * (k * np.log(p) + (n-k) * np.log(1-p) + np.log(comb(n,k)))
```

例としてコイン投げを想定して、10回投げて6回表が出た場合を考える。このときの尤度関数は
```python
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.special import comb

fig, ax = plt.subplots(1,2,figsize=(10,3))

x = np.linspace(0,1,100)
# 尤度計算
L = [likelihood_bernoulli(ip,10,6) for ip in x]
ax[0].plot(x, L, color="black")

# 対数尤度計算
NLL = [negative_loglikelihood_bernoulli(ip,10,6) for ip in x]
ax[1].plot(x, NLL, color="black")

# --- 以下プロットの調整 --- 
ax[0].set_xlim(0,1)
ax[0].set_ylim(0)
ax[0].set_xlabel("$ p $", fontsize=12)
ax[0].set_ylabel("$ L(p) $", fontsize=12)

ax[1].set_xlim(0,1)
ax[1].set_ylim(0)
ax[1].set_xlabel("$ p $", fontsize=12)
ax[1].set_ylabel("$ -\logL(p) $", fontsize=12)

plt.show()
```

![](./img/likelihood/20220920-104340.png)

で表される。尤度関数（左図）が最大となる場所が $p=0.6$ になっているので、直感的にな成功確率 0.6 とも相反しない。
ただし最尤法で求めた $p=0.6$ が真値（真の表が出る確率）であるかどうかは分からないことに留意。

また、負の対数尤度関数（右図）は同様に $p=0.6$ で極小値を取っている。数値計算の際にはこちらの負の対数尤度（negative log-likelihood）を用いて最小値を求める形で、最尤推定量を計算することがある。

### 最尤推定量

`scipy.optimize.minimize` を利用して、最尤推定量を求める。`minimize` を使用するために関数を（試行回数、成功回数を引数に取らないように）少し修正して使用する。ソルバーとしては `Nelder-Mead` 法を用いているが、最適化計算についての詳細は別記事でまとめることとしよう。

```python
from scipy.optimize import minimize

def func(p):
    n = 10
    k = 6
    return -1 * (k * np.log(p) + (n-k) * np.log(1-p) + np.log(comb(n,k)))

res = minimize(func, x0=[0], method="Nelder-Mead")

> print(res)
l_simplex: (array([[0.6      ], [0.5999375]]), array([1.38300914, 1.38300922]))
           fun: 1.3830091393750967
       message: 'Optimization terminated successfully.'
          nfev: 46
           nit: 23
        status: 0
       success: True
             x: array([0.6])
```

結果を見て（`x`）分かる通り、最尤推定量を数値計算で求めても $p=0.6$ とちゃんと求まっていることが分かる。
最適化計算を可視化すると以下のようになる（以下の内容は単純に gif を作成する、というだけのもの）。
```python
# ------------ 
# 最適化計算
# callback を使用して、計算過程の値を取得する
# ------------ 
def callback(path):
    def minimize_cb(xk):
        path.append(xk)
    return minimize_cb

path = []
res = minimize(func, x0=[0], method="Nelder-Mead", callback=callback(path))
path = np.array(path)

# ------------ 
# gif 作成
# ------------ 
fig, ax = plt.subplots(1,1,figsize=(5,3))

x = np.linspace(0,1,100)
NLL = [negative_loglikelihood_bernoulli(ip,10,6) for ip in x]

ax.plot(x, NLL, color="black")
ax.set_xlim(0,1)
ax.set_ylim(0)
ax.set_xlabel("$ p $", fontsize=12)
ax.set_ylabel("$ -\logL(p) $", fontsize=12)

# mp4 --> gif
def animate(i):
    ax.plot(path[i], func(path[i]), marker="o", color="orange")
    
an = animation.FuncAnimation(fig, animate, frames=len(path)) 
an.save("bernoulli.mp4")

clip = VideoFileClip("bernoulli.mp4")
clip.write_gif("bernoulli.gif")
```


## 正規分布

母平均 $\mu$、母分散 $\sigma$ のガウス分布 $N(\mu,\sigma)$ に独立に従う事象を $X=(X_1,X_2,...,X_n)$のデータとして観測した場合の同時確率密度関数は
$$
L(\bm{\theta}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(x_i-\mu)^2}{2\sigma^2} \right)
$$
である。対数尤度関数は
$$
\log L(\bm{\theta}) = - n\log \sqrt{2\pi}\sigma - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2
$$
である。二項分布のときのように、尤度関数を計算する関数を用意してパラメータを当てはめて計算してもよいが、
`scipy.stats.norm` の関数を使用することで、もう少し簡単に尤度関数を計算できる。

例として、正規分布 $N(0,1)$ に従うデータを10点観測した場合の尤度関数を考える。