# Welford Algorithm 

The Welford algorithm or one-pass computation for mean and variance is extended to compute the higher moments. Following is the step-by-step derivation of the moments.

Let $x_i$ be the $i$-th data point. 
Let $M_1^{(n)}$ be the running mean after $n$ samples. 
Let $M_k^{(n)} = \sum_{i=1}^n (x_i - M_1^{(n)})^k$ be the sum of the $k$-th powers of differences from the mean. 
When a new point $x_n$ arrives, we define the difference from the old mean: 
$$\delta = x_n - M_1^{(n-1)}$$ 
And the amount the mean will shift by: 
$$\delta_n = \frac{\delta}{n}$$ 

To update the sums, we need to know two things:
1. How the old points shift: For any previous point ($i < n$), its distance to the new mean is its distance to the old mean, minus the mean shift: $$x_i - M_1^{(n)} = (x_i - M_1^{(n-1)}) - \delta_n$$
2. How the new point fits in: The distance of the new point $x_n$ to the new mean is: $$x_n - M_1^{(n)} = x_n - (M_1^{(n-1)} + \delta_n) = \delta - \delta_n = \delta \left(1 - \frac{1}{n}\right) = (n-1)\delta_n$$

With these two shifts established, we can derive the moments by expanding the binomials $(A - B)^k$.

## Deriving $M_1^{(n)}$ (The Mean)

The definition of the mean is the sum of all points divided by $n$. We separate the old points from the new point $x_n$:
$$M_1^{(n)} = \frac{1}{n} \left( \sum_{i=1}^{n-1} x_i + x_n \right)$$
Since $\sum_{i=1}^{n-1} x_i = (n-1)M_1^{(n-1)}$, we substitute:
$$M_1^{(n)} = \frac{1}{n} \left( (n-1)M_1^{(n-1)} + x_n \right)$$
$$M_1^{(n)} = M_1^{(n-1)} + \frac{x_n - M_1^{(n-1)}}{n}$$
$$M_1^{(n)} = M_1^{(n-1)} + \delta_n$$

## Deriving $M_2^{(n)}$ (Sum of Squares)

We split the sum of squares into the old points (using Shift 1) and the new point (using Shift 2):
$$M_2^{(n)} = \sum_{i=1}^{n-1} \left[ (x_i - M_1^{(n-1)}) - \delta_n \right]^2 + \left[ (n-1)\delta_n \right]^2$$
Expand the binomial $(a - b)^2 = a^2 - 2ab + b^2$:
$$M_2^{(n)} = \sum_{i=1}^{n-1} (x_i - M_1^{(n-1)})^2 - 2\delta_n \sum_{i=1}^{n-1} (x_i - M_1^{(n-1)}) + \sum_{i=1}^{n-1} \delta_n^2 + (n-1)^2 \delta_n^2$$
Now, simplify the terms:
* The first term is exactly $M_2^{(n-1)}$.
* The second term is $0$, because the sum of differences from a mean is always zero.
* The third term is simply $(n-1)\delta_n^2$.

$$M_2^{(n)} = M_2^{(n-1)} - 0 + (n-1)\delta_n^2 + (n-1)^2 \delta_n^2$$
Factor out $(n-1)\delta_n^2$:
$$M_2^{(n)} = M_2^{(n-1)} + (n-1)\delta_n^2 (1 + n - 1) = M_2^{(n-1)} + n(n-1)\delta_n^2$$
Because $n\delta_n = \delta$, we can rewrite $n \cdot \delta_n \cdot \delta_n$ as $\delta \cdot \delta_n$:
$$M_2^{(n)} = M_2^{(n-1)} + \delta \cdot \delta_n \cdot (n-1)$$

## Deriving $M_3^{(n)}$ (Sum of Cubes)

We follow the exact same logic, using Shift 1 and Shift 2, but expand the cubic binomial $(a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3$:
$$M_3^{(n)} = \sum_{i=1}^{n-1} \left[ (x_i - M_1^{(n-1)}) - \delta_n \right]^3 + \left[ (n-1)\delta_n \right]^3$$
$$M_3^{(n)} = \sum_{i=1}^{n-1} (x_i - M_1^{(n-1)})^3 - 3\delta_n \sum_{i=1}^{n-1} (x_i - M_1^{(n-1)})^2 + 3\delta_n^2 \sum_{i=1}^{n-1} (x_i - M_1^{(n-1)}) - \sum_{i=1}^{n-1} \delta_n^3 + (n-1)^3 \delta_n^3$$

Substitute the known sums:
* Term 1 is $M_3^{(n-1)}$.
* Term 2 is $3\delta_n M_2^{(n-1)}$.
* Term 3 is $0$ (sum of differences from mean).
* Term 4 is $(n-1)\delta_n^3$.

$$M_3^{(n)} = M_3^{(n-1)} - 3\delta_n M_2^{(n-1)} - (n-1)\delta_n^3 + (n-1)^3 \delta_n^3$$
Factor $\delta_n^3$ from the last two terms:
$$\delta_n^3 \left[ (n-1)^3 - (n-1) \right] = \delta_n^3 (n-1) \left[ (n-1)^2 - 1 \right] = \delta_n^3 (n-1)(n^2 - 2n) = \delta_n^3 \cdot n(n-1)(n-2)$$
Substitute $n\delta_n = \delta$:
$$M_3^{(n)} = M_3^{(n-1)} - 3\delta_n M_2^{(n-1)} + \delta \cdot \delta_n^2 \cdot (n-1)(n-2)$$

## Deriving $M_4^{(n)}$ (Sum of Fourth Powers)

We expand the quartic binomial $(a - b)^4 = a^4 - 4a^3b + 6a^2b^2 - 4ab^3 + b^4$:
$$M_4^{(n)} = \sum_{i=1}^{n-1} \left[ (x_i - M_1^{(n-1)}) - \delta_n \right]^4 + \left[ (n-1)\delta_n \right]^4$$
$$M_4^{(n)} = M_4^{(n-1)} - 4\delta_n M_3^{(n-1)} + 6\delta_n^2 M_2^{(n-1)} - 0 + (n-1)\delta_n^4 + (n-1)^4 \delta_n^4$$
Factor the final $\delta_n^4$ terms:
$$\delta_n^4 \left[ (n-1)^4 + (n-1) \right]$$
Factor out $(n-1)$:
$$\delta_n^4 (n-1) \left[ (n-1)^3 + 1 \right]$$
Expand $(n-1)^3 = n^3 - 3n^2 + 3n - 1$:
$$\delta_n^4 (n-1) (n^3 - 3n^2 + 3n - 1 + 1) = \delta_n^4 (n-1)(n^3 - 3n^2 + 3n)$$
Factor out $n$:
$$\delta_n^4 \cdot n(n-1)(n^2 - 3n + 3)$$
Substitute $n\delta_n = \delta$:
$$\delta \cdot \delta_n^3 \cdot (n-1)(n^2 - 3n + 3)$$
Putting it all together yields the final formula:
$$M_4^{(n)} = M_4^{(n-1)} - 4\delta_n M_3^{(n-1)} + 6\delta_n^2 M_2^{(n-1)} + \delta \cdot \delta_n^3 \cdot (n-1)(n^2 - 3n + 3)$$

## Skewness and excess kurtosis:

We can compute skewness and excess kurtosis from the raw accumulators ($M_2$, $M_3$, $M_4$) that we built up in the loop.

### The Conditions: (m2 > 0) & (count >= 3)

Before doing the math, the code checks two things to prevent errors:
* m2 > 0: $M_2$ represents the variance (spread) of the data. If $M_2 = 0$, it means all the data points are exactly the same. If there is no spread, skewness and kurtosis are mathematically undefined. Checking for > 0 prevents a division by zero error.
* count >= 3 (or 4): To calculate skewness, you need at least 3 data points. To calculate kurtosis, you need at least 4. If you try to calculate them with fewer points, the results are meaningless.

If the conditions are met, the code executes the standard statistical formulas using the accumulators.

#### For Skewness:
(np.sqrt(count) * m3) / (m2_safe ** 1.5)
This translates to the population skewness formula:
$$Skewness = \frac{\sqrt{n} \cdot M_3}{M_2^{1.5}}$$

##### Note
m2_safe is used instead of m2 because taking a fractional power (** 1.5) of a negative number in Python results in a NaN. Even though $M_2$ is mathematically always positive, tiny floating-point errors can sometimes make it slightly negative (e.g., -0.0000000001). m2_safe clips those tiny negative errors to exactly 0.

#### Excess Kurtosis:
(count * m4) / (m2_safe * m2_safe) - 3.0
This translates to:
$$Excess\ Kurtosis = \frac{n \cdot M_4}{M_2^2} - 3$$

##### Note: 
We subtract 3 to calculate the excess kurtosis. A standard normal distribution has a raw kurtosis of 3. Subtracting 3 shifts the baseline so that a perfectly normal distribution has a kurtosis of exactly 0, making it easier to interpret.

#### The Fallback

##### nan_template
If the conditions are not met (for example, you only have 2 data points, or all your data points are zeros), the xr.where function returns nan_template.
This is simply an array of the same shape filled with NaN (Not a Number). It safely tells you "there is no valid result for this coordinate" without crashing the entire script.











