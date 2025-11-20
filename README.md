# Portfolio-Optimization-Model

This is a simple portfolio optimization tool built via a monte carlo simulation based on Markowitz Portfolio Theory.

Further improvements: 
- to create one based on the Black Litterman Model


# to do 
-Portfolio constructor (optimization engine:
                        -MPT for now
                        )
-Backtest class


Steps (Rough):              HSLMI + top 40 ranked mom
1. Pull daily adj close for stocks from API 
- window size ? this will depend on how often i plan to rebalance


2. Decide on a momentum metric, filter top 30 stoocks
- used a blended score

3. Calculate mean and std of returns for each stock

4. Randomise weights and simulate montecarlo, plot eff frontier
-instead of MVO, 4 weighting methods,:
eql w
scvaling by a slow moving risk tilt, inverse vol,
scale by 1st pca
MVO

5. Get best combi of weights

# covariance matrix shrinkage