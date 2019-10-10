## PruningProblems
### 1.  the ‘typical’ error of a tree trained on m data points
Use the function `getdata(m)` generate `m` data points. Use the function `decisionTree(data).fit()` to fit a tree. Use the function `decisionTree(data).predict(X)` and `decisionTree(data).getErrRate(res, Y)` to get error rate.

Results are shown below: (Fig. 1)
![enter image description here](https://lh3.googleusercontent.com/AzzgRcNaHY2eqHe6neryuZuCuW-dEKAYeJ6nmZ1tSAVn5l7TrCdjmZah587vv5x5j7SQA02DlhSb)
Fig.1 the "typical" error of a tree trained on `m` data points.

As `m` increases, the error rate are decreasing. Nevertheless, the rate of decreasing is slowing down.

It is reasonable that if we have more samples, we will know more accurately about the whole table. In another word, the error rate will decrease. Also, when we have enough data points, a small number of new data points will only provide limited information by the notion of the marginal value.

Another approach is to consider the sample complexity. Notice that a decision tree without pruning will try it best to make $err_{train}$ down to $0$. Therefore, $|err(f) - err_{train}(f)| \approx err(f)$. In this case, if we want $err(f) < \epsilon$, we need $m \ge \frac{1}{\epsilon^2} 2^k$. Namely, $err(f) \varpropto \frac{1}{\sqrt m}$.

### 2. How much data would you need, typically, to avoid ﬁtting on this noise?
Here is the results: (Fig. 2)

![enter image description here](https://lh3.googleusercontent.com/h8GEq5yO_aRlBTFl9fEwoRxktIkbaA_s04-xr15ClBxxRiYxtYYsBM7rMPV5iornBaJkUxLReskS)
Fig.2 the number of irrelevant variables with different `m`

When `m` is less than about 1600, the tree is not large enough such that it is probable to describe all data points with a few number of features. In this case, it is likely to exclude many variables both relevant and irrelevant.

When the tree is large enough, it seems that the number of irrelevant variables slightly decreases. Here is the result when `m > 10000`. (Fig. 3)

![enter image description here](https://lh3.googleusercontent.com/R4BQnszv4VaZp5yJEVajtwdGxt4MDw4v0ugmpsOvxZd363rYIUkEomYYRl6T0GtxOfR5GuD3WGrX)
Fig.3 the number of irrelevant variables with different `m`

In general, the number of irrelevant variables does decrease as `m` gose up. But it is extreamly slow.

`tree1M.txt` is a tree trained with 1 million data points. Notice that all irrelevant variables appear at the last branch node, which is at least with the depth of 11. In this case, even if we have about $2^{20}$ data points, these nodes will have only about $500$ data points. (Actually, the real number is about $16$, which is much less than $500$ because of the results of Question 6.) In this case, it is extremely likely that a irrelevant variables has a higher information gain just by coincidence.

Also, when there are more irrelevant variables, say from $m_{15}$ to $m_{40}$ are irrelevant. the decrease is more notable. (Fig. 4)

![enter image description here](https://lh3.googleusercontent.com/zphO5kjlEtHti_XQkcZSmDaowmeEfEusEZy73BP2NqOWDSzvXpRIYl00Wx_bXcPMYZ_ZS7TZl9rQ)
Fig.4 the number of irrelevant variables when there are 26 irrelevant variables.

In conclusion, 1 million data points are not enough to avoid fitting on this noise. Personally, I could not predict how much data is enough. But I guess that hopefully $2^{50}$ data points might be enough.

### 3. a data set of size m = 10000
#### a. Pruning by Depth
Results are shown below: (Fig. 5)

![enter image description here](https://lh3.googleusercontent.com/6nCm3rqsdLcZgRhMd7xqHH4KbxBlPe3__Im7VQRDMp_Sl6-oqvUm0xAPmclvAJ10ZgkFr7Bac0jb)
Fig.5 the errors with different pruning depth.

Notice that when `depth < 9`, the test error is greater than $3\%$, and keeps decreasing. When `depth == 9`, we have the smallest $|error_{train} - error_{test}|$.
Therefore, 9 is a good threshold depth.

#### b. Pruning by Sample Size
Results are shown below: (Fig. 6)
![enter image description here](https://lh3.googleusercontent.com/9THYDeTIk8EEEqxoYCo3OXhKlMLIoxs1Mhj7o8tJ8xNawO_4LSp7q-hC4wCUM1YFhBDYUbqmiXer)
FIg.6 the errors with different pruning size.

Notice that when `size > 16`, the test error is greater than $4\%$, and keeps decreasing. When `depth == 16`, we have the smallest $|error_{train} - error_{test}|$.
Therefore, 16 is a good threshold size.

#### c. Pruning by Signiﬁcance
Results are shown below: (Fig. 6)
![enter image description here](https://lh3.googleusercontent.com/08m8JrMMaynvaqGgJeyZOudA5WgltBX6Kqli8mAxHfj8o75XfQWxs1cSwdcvOcitHZ9-2snDOA7f)
FIg.6 the errors with different pruning size.

Notice that when `score < 10.828`, the test error is greater than $3\%$, and keeps decreasing. When `score == 10.828`, we have the smallest $|error_{train} - error_{test}|$.
Therefore, 10.828 is a good threshold score.
