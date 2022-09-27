# 最適化計算

SciPy の minimize を用いた最適化計算をまとめる。

# はじめに

関数最適化とは与えられた制約上の中で、その関数の最大値または最小値（or 極小値・極大値）を求める計算のこと。最尤法における尤度関数の最大化や、機械学習における損失関数の最小化など、最適化計算には何かと出会う場面が多い。

`scipy.optimize.minimize` の使い方を簡単に、2次関数を例にしてまとめる。

- func : 最適化したい関数
- x0：初期値
- args：
- method：
- jac：
- hess：
- hessp：
- bounds：
- constraint：
- tol：
- options：
- callback：

返り値：
`OptimizerResult` オブジェクト。


## 例. $y=(x+1)^2$ の最小値を求める

```python
def func(x):
    return (x+5)**2

x0 = np.array([20])
res = minimize(func, x0=x0, method="Nelder-Mead", tol=1e-6)
```
計算結果の確認
```python
$ print(res)
 final_simplex: (array([[-5.        ],
       [-5.00000095]]), array([0.00000000e+00, 9.09494702e-13]))
           fun: 0.0
       message: 'Optimization terminated successfully.'
          nfev: 56
           nit: 28
        status: 0
       success: True
             x: array([-5.])
```
最後の `res.x` が最適化計算の結果であり、正しく $y=(x+5)^2$ の最小値を求められていることが分かる。


# 各種手法（ソルバー）

## Nelder-Mead（ネルダー・ミード法）

## Powell
 
## CG
 
## BFGS
 
## Newton-CG
 
## L-BFGS-B
 
## TNC 
 
## COBYLA
 
## SLSQP 
 
## trust-constr
 
## dogleg 
 
## trust-ncg 
 
## trust-exact
 
## trust-krylov
