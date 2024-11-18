[some previous notes](./old%20Data-100.md)

# Lab and Hw
## NumPy
The core of NumPy is the array. Like Python lists, arrays store data; however, they store data in a more efficient manner. In many cases, this allows for faster computation and data manipulation.

However it cannot store items of different data types like lists in python. 

Arrays are also useful in performing *vectorized operations*. Given two or more arrays of equal length, arithmetic will perform element-wise computations across the arrays.

Python is slow, but array arithmetic is carried out very fast, even for large arrays.

The `@` operator multiplies NumPy matrices or arrays together

## Path 
In Python, a `Path` object represents the filesystem paths to files (and other resources). The `pathlib` module is effective for writing code that works on different operating systems and filesystems.

To check if a file exists at a path, use `.exists()`. To create a directory for a path, use `.mkdir()`. To remove a file that might be a [symbolic link](https://en.wikipedia.org/wiki/Symbolic_link), use `.unlink()`.

This function creates a path to a directory that will contain data files. It ensures that the directory exists (which is required to write files in that directory), then proceeds to download the file based on its URL.

The benefit of this function is that not only can you force when you want a new file to be downloaded using the `force` parameter, but in cases when you don't need the file to be re-downloaded, you can use the cached version and save download time.

## Pandas 

### agg

`groupby.agg`是Pandas库中DataFrame和Series对象提供的一个强大且灵活的方法，用于对数据进行分组和聚合操作。它允许你基于一个或多个键（列）对数据进行分组，并对每个分组应用一个或多个聚合函数。

`grouped = dataframe.groupby('column_name').agg(functions)`

- `dataframe`: 要进行分组和聚合操作的DataFrame。
- `column_name`: 用于分组的列名，可以是单个列名或列名列表。
- `functions`: 要应用的聚合函数，可以是内置函数（如`sum`, `mean`, `min`, `max`等）、用户自定义函数或函数字典。

## Kernel Density Estimation 
Kernel density estimation is used to estimate a probability density function (i.e. a density curve) from a set of data. Just like a histogram, a density function's total area must sum to 1.

KDE centrally revolves around this idea of a "kernel". A kernel is a function whose area sums to 1. The three steps involved in building a kernel density estimate are:
1. Placing a kernel at each observation
2. Normalizing kernels so that the sum of their areas is 1
3. Summing all kernels together

The end result is a function, that takes in some value `x` and returns a density estimate at the point `x`.

## Matplotlib and Seaborn Table of Common Function 
| Function                                 | Description                                                                                                                                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `plt.plot(x, y)`                         | Creates a line plot of `x` against `y`                                                                                                                                                                                                                                                     |
| `plt.title(name)`                        | Adds a title `name` to the current plot                                                                                                                                                                                                                                                    |
| `plt.xlabel(name)`                       | Adds a label `name` to the x-axis                                                                                                                                                                                                                                                          |
| `plt.ylabel(name)`                       | Adds a label `name` to the y-axis                                                                                                                                                                                                                                                          |
| `plt.scatter(x, y)`                      | Creates a scatter plot of `x` against `y`                                                                                                                                                                                                                                                  |
| `plt.hist(x, bins=None)`                 | Creates a histogram of `x`; `bins` can be an integer or a sequence                                                                                                                                                                                                                         |
| `plt.bar(x, height)`                     | Creates a bar plot of categories `x` and corresponding heights `height`                                                                                                                                                                                                                    |
| `sns.histplot(data, x, y, hue, kde)`     | Creates a distribution plot; `data` is a DataFrame; `x`, `y` are column names in `data` that specify positions on the x and y axes; `hue` is a column name in `data` that adds subcategories to the plot based on `hue`; `kde` is a boolean that determines whether to overlay a KDE curve |
| `sns.lineplot(data, x, y, hue)`          | Creates a line plot                                                                                                                                                                                                                                                                        |
| `sns.scatterplot(data, x, y, hue, size)` | Creates a scatter plot; `size` is a vector that contains the size of point for each subcategory based on `hue`                                                                                                                                                                             |
| `sns.kdeplot(x, y)`                      |  Creates a kernel density estimate plot; `x`, `y` are series of data that indicate positions on the `x` and `y` axis                                                                                                                                                                       |
| `sns.jointplot(x, y, data, kind)`        | Creates a joint plot of 2 variables with KDE plot in the middle and a distribution plot for each variable on the sides; `kind` determines the visualization type for the distribution plot, can be `scatter`, `kde` or `hist`                                                              |
## Introduction to sklearn
Another way to fit a linear regression model is to use **scikit learn**, an industry standard package for machine learning applications. `sklearn` is often faster and more robust than analytical/scipy-based computation methods we've used thus far
# Class 
## Pandas 
### Indexing with `loc`, `iloc`, and `[]`

One of the most basic tasks for manipulating a DataFrame is to extract rows and columns of interest. 

The Pandas library has a lot of "syntactic sugar": Methods that useful and lead to concise code, but not absolutely necessary for the library to function.

Fundamental `loc` selects items by label 
Arguments to `loc` can be 
- A list 
- A slice (syntax is inclusive of the right hand side of the slice)
- A single value 
you can omit the second argument if you want all columns. If you want all rows, but only some columns, you can use : for the left argument.

Fundamental `iloc` selects items by number
Arguments to `iloc` can be
- A list.
- A slice(syntax is exclusive of the right hand side of the slice).
- A single value.

`loc`:
- Safer: If the order of columns gets shuffled in a public database, your code still works
- Legible: Easier to understand what means 
`iloc`:
- If you have a DataFrame of movie earnings sorted by earnings, can use `iloc` to get the median earnings for a given year.

Selection operators:
- `loc` selects items by label. First argument is rows, second argument is columns.
- `iloc` selects items by number. First argument is rows, second argument is columns.
- `[]` only takes one argument, which may be:
	- A slice of row numbers.
	- A list of column labels.
	- A single column label.

In practice, we'll use `[]` a lot, especially when selecting columns 
- Syntax for `[]` is more concise for many commob uses cases! 
- `[]` is much more common in real world practice than `loc` 

### DataFrames, Series, and Indices 

There are three fundamental data structures in pandas:
- DataFrame: 2D data tabular data 
- Series: 1D data.
- Index: A sequence of row labels 

We can think of a DataFrame as a collection of Series that all share the same Index 

An Index can also:
- Be non-numeric 
- Have a name 
The row labels that constitute an index do not have to be unique. But the column names in Pandas are almost always unique.

### Conditional Selection 

Another input type supported by `loc` and `[]` is the boolean array. 

Boolean array selection is a useful tool, but can lead to overly verbose code for complex conditions.
Pandas provides many alternatives:
- `.isin`
- `.str.startwith`
- `.query`
- `.groupby.filter` 

### Handy Utiliry Functions 

Pandas Series and DataFrames support a large number of operations, including mathematical operations, so long as the data is numerical.

In addition to its rich syntax for indexing and support for other libraries (numpy, built-in functions), Pandas provides an enormous number of useful utility functions. 
- size/shape
- describe
- sample
- value_counts
- uniques
- sort_values

If you want a DataFrame consisting of a random selection of rows, you can use the sample method.
- By default, it is without replacement. Use `replace = true` for replacement.
- Naturally, can be chained with other methods and operators.

The `Series.value_counts` method counts the number of occurrences of a each unique value in a Series.
- Return value is also a Series. 

The `Series.unique` method returns an array of every unique value in a Series.

The `DataFrame.sort_values `and `Series.sort_values` methods sort a DataFrame (or Series).

## Exploratory Data Analysis(EDA)

The process of **transforming**, **visualizing**, and **summarizing** data to:
- Build/confirm understanding of the data and its provenance
- Identify and address potential issues in the data
- Inform the subsequent analysis
- Discover potential hypothesis … (be careful…)

EDA is an *open-ended* analysis
- Be willing to find something surprising! 

### Key Data Properties to consider in EDA 

 **What should we look for**
 - **Structure** the "shape" of a data file
 - **Granularity** how fine/coarse is each datum
 - **Scope** how (in)complete is the data
 - **Temporality** how is the data situated in time 
 - **Faithfulness** how well does the data capture "reality"

#### Structure 

##### File Format  
We prefer rectangular data for analysis 
- Regular structures are easy manipulate and analyze
- A big part of data cleaning is about transforming data to be more recangular

Two kinds of rectangular data: **Tables** and **Matrices**
**Tables**
- Named columns with different types
- Manipulated using data transformation languages 
**Matrices**
- Numeric data of the same type
- Manipulated using linear algebra

**CSV** Comma Separated Values
CSV is a very common table file format
- Records(rows) are delimited by a newline
- Fields are delimited by commas 

**TSV** Tab Separated Values
Another common table file format 
- Records are delimited by a newline
- Fields are delimited by `'\t'`

**JSON**  JavaScript Object Notation 
A less common table file format 
- Very similar to Python dictionaries 
- Strict formatting "quoting" address some issues in CSV/TSV
- Can save metadata(data about the data) alongwith records in the same file 
Issues 
- Not retangular
- Each record can have different fields 
- Nesting means records can contain tables - complicated 

##### Variable Type 
All data is composed of records. Each record has a set of *variables*(aka *fileds*) 
- Tabular: Records == Rows, Variables == Columns 
- Non-Tabular: Create Records and wrangle into tabular data 

Variables are defined by their type 
- Storage type 
- Feature type: conceptual notion of the information 

![[Pasted image 20240810135337.png]]

##### Multipel files 
Sometimes you data comes in muliple files 

**Primary Key**: the column or set of columns in a table that determine the values of the remaining columns 
- Primary keys are unique

**Foreign keys**: the column or sets of columns that reference primary keys in other tables 
#### Faithful 

##### Do I trust this data? 

Does my data contain **unrealistic or “incorrect” values**?
- Dates in the future for events in the past
- Locations that don’t exist
- Negative counts
- Misspellings of names
- Large outliers

Does my data violate **obvious dependencies**?
- E.g., age and birthday don’t match

Was the data **entered by hand**?
- Spelling errors, fields shifted …
- Did the form require all fields or provide default values?

Are there obvious signs of **data falsification**?
- Repeated names, fake looking email addresses, repeated use of uncommon names or fields.

##### Not be faithful and Solutions 

**Truncated data**
- Early Microsoft Excel limits: 65536 Rows, 255 Columns
- Soln: be aware of consequences in analysis ⇒ how did truncation affect sample?

**Time Zone Inconsistencies**
- Soln 1: convert to a common timezone (e.g., UTC) 
- Soln 2: convert to the timezone of the location – useful in modeling behavior.

**Duplicated Records or Fields**
- Soln: identify and eliminate (use primary key) ⇒ implications on sample?

**Spelling Errors**
- Soln: Apply corrections or drop records not in a dictionary ⇒ implications on sample?

**Units not specified or consistent**
- Solns: Infer units, check values are in reasonable ranges for data  

##### Missing Data 

**Drop records** with missing values
- Probably most common
- Caution: check for biases induced by dropped values
	- Missing or corrupt records might be related to something of interest

**Imputation**: Inferring missing values
- Average Imputation: replace with an average value   
	- Which average?  Often use closest related subgroup mean.
- Hot deck imputation: replace with a random value 
	- Choose a random value from the subgroup and use it for the missing value.

**Other Suggestions**
1. **Drop** missing values but check for **induced bias**
2. Directly **model missing values** during future analysis 

### EDA steps 
1. Understand what each record, each feature represents
2. Hypothesize why these values were missing, then use that knowledge to decide whether to drop or impute missing values.

## Regular Expressions 
[regex101: build, test, and debug regex](https://regex101.com/)

With text data, wrangling is upfront and requires new tools: **Python string manipulation** and **regular expressions**.

Why work with text?
- **Canonicalization**: Convert data that has more than one possible presentation into a standard form
- **Extract** information into a new feature. 

Python string functions:
- Are very brittle! Requires maintenance.
- Have limited flexibility.

An alternate approach is to use a **regular expression**
- Implementation provided in the Python re library and the pandas str accessor

### Syntax

A **formal language** is a set of strings, typically descirbed implicitly.
A **regular languag**e is a formal language that can be described by a **regular expression**.

A **regular expression**("**regex**") is a sequence of characters that specifies a search pattern.

![[Pasted image 20240811184429.png]]

![[Pasted image 20240811185450.png]]

![[Pasted image 20240811200944.png]]

![[Pasted image 20240811201101.png]]

### Regex in Python and Pandas
`re.sub(pattern, repl, text)`
Returns text with all instances of pattern replaced by repl 

pattern is a **raw string** `r"..."`

`ser.str.replace(pattern, repl, regex = True)`
Returns Series with all instance of pattern in Series ser replaced by repl 

When specifying a pattern, we strongly suggest using raw strings.
- A raw string is created using `r""` or `r''` instead of just  `""` or `''`
- The exact reason is a bit tedious
	- Rough idea: Regular expressions and Python strings both use `\` as an escape character.
	- Using non-raw strings leads to uglier regular expressions 

`re.findall(pattern, text)`
Return a list of all matches to pattern

`ser.str.findall(pattern)`
Returns a Series of lists 

Earlier we used parentheses to specify the order of operations
Parenthesis have another meaning:
- Every set of parentheses specifies a match/capture group
- In Python, matches are returned as tuples of groups.

### Limitations 
Writing regular expressions is like writing a program
- Need to know the syntax well 
- Can be easier yo write than to read 
- Can be difficult to debug 

Regular expressions sometimes jokingly referred to as s **write on language**

Regular expressions are terrible be at certain types of problems:
- For parsing a hierachical structure such as JSON not regex! 
- Complex features 
- Counting
- Complex properties 

Regular expressions are decent at wrangling text data. 

## Visualization 
### Plots

A **distribution** describes the frequency at which values of a variable occur
- All values must be accounted for once, and only once.
- The total frequencies must add up to 100%, or to the number of values that we're observing

**Bar Plots** are the most common way of displaying the distribution of a qualitative(categorical) variable.

An overlaid "rug plot" lets us see the distribution of data points within each bin 

Histograms allow us to assess a distribution by their shape.

If a distribution has a long right tail, we call it **skewed right**
- Mean is typically to the right of the median 
	- Think of the mean as the "balancing point" of the density.
- If the tail is on the left, we say the data is skewed left.
- Our distribution can be also symmetric, when both tails are of equal size.

A mode of a distribution is a local or global maximum
- A distribution with a single clear maximum is called unimodal
- Distributions with two modes are called bimodal
	- More than two: multimodal
- Need to distinguish between modes and random noise.

Instead of a discrete histogram, we can visualize what a continuous distribution corresponding to that same histogram could look like...
- The smooth curver drawn on top of the histogram here is called a density curve.

For a quantitative variable:
- First or lower quartile: 25th percentile
- Second quartile: 50th percentile(median)
- Third or upper quartile: 75 percentile

**Interquartile range(IQR)** measures spread.
- IQR = third quartile - first quartile

**Box plots** summarize several characteristics of a numerical distribution. They visualize:
- Lower quartile
- Median  
- Upper quartile 
- "Whiskers", placed at lower quartile minus 1.5\*IQR and upper quartile plus 1.5\*IQR 
- Outliers, which are defined as being further than 1.5\*IQR from the extreme quartiles. Arbitrary definition 
- We also lose a lot of information 
![[Pasted image 20240812104233.png]]

Violin plots are similar to box plots, but also show somoothed density curves
- The "width" of our "box" now has meaning 
- The three quartiles and "whisker" are still present - look closely

### KDE 
In general, we smooth if we want to focus on **general sturcture** rather than individual observation

Kernel Density Estimation is used to estimate a **probability density function**(or **density curve**) from a set of data
To create a KDE:
- Place a kernel at each data point
- Normalize kernels so that total area = 1
- Sum all kernels together.
To generate a curve we need to choose a kernel and bandwidth 

A kernel is a valid density function.
- Must be non-negative for all inputs
- Must integrate to 1

The most common kernel is the Gaussian kernel 
$$
K_\alpha(x,x_i) = \frac{1}{\sqrt{2\pi\alpha^2}}e^{-\frac{(x-x_i)^2}{2\alpha^2}}
$$
where $x$ represents any input, and $x_i$ represents the ith observed value.$\alpha$ is the bandwidth parameter, It controls the smoothness of our KDE.

**Bandwidth** is analogous to the width of each bin a histogram 
- As $\alpha$ increase, the KDE becomes more smooth.
- Large $\alpha$ KDE is simpler to understand, but gets rid of potentially important distributional information.

general "KDE formula" function:
$$
f_\alpha(x) = \frac{1}{n}\sum^{n}_{i=1}K_\alpha(x,x_i)
$$

### Visualization Theory 

- **Qualitative**: Choose a qualitative scheme that makes it easy to distinguish between categories.
- **Quantitative**: Choose a color scheme that implies magnitude.

If the data progresses from low to high, use a **sequential** scheme where lighter colors are for more extreme values.
If low and high values deserve equal emphasis, use a **diverging** scheme where lighter colors represent middle values.

- Don't use pie charts
- Avoid area charts
- Avoid word clouds 
Stacked bar charts, histograms, and area charts are hard to read because the baseline moves.

A publication-ready plot needs:
- Informative title (takeaway, not description).
	- “Older passengers spend more on plane tickets” instead of “Scatter plot of price vs. age”.
- Axis labels.
- Reference lines, markers, and labels for important values.
- Legends, if appropriate.
- Captions that describe the data.

Captions should be:
- Comprehensive and self-contained.
- Describe what has been graphed.
- Draw attention to important features.
- Describe conclusions drawn from graph.

**Summary**
Some key ideas from today:
- KDEs are not magic! They’re just copies of a Gaussian curve added together.
- Choose appropriate scales.
- Choose colors and markings that are easy to interpret correctly.
- Condition in order to make comparisons more natural.
- Add context and captions that help tell the story.
- Transforming our data can linearize relationships.
	- Helpful when we start linear modeling next lecture.
- More generally – reveal the data!
	- Eliminate anything unrelated to the data itself – “chart junk.”
	- It’s fine to plot the same thing multiple ways, if it helps fit the narrative better.

## Introduction to Modeling, SLR 
A model is an **idealized representation** of a system

Purposes:
- To understand **complex phenomena** occurring in the world we live in.
- To make **accurate predictions** about unseen data
Most of the time we want to strike a balance between interpretability and accuracy. 

**Physical models**: Laws that govern how the world works.
**Statiscal models**: Relationships between variables found through data and statistical analysis

Simple Linear Regression Model (SLR):
$$
\hat{y} =a +bx
$$
SLR is a **parametric model**, meaning we choose the "best" parameters for slope and intercept based on data.

$\hat{y}= \hat{a}+\hat{b}x$ The "best" linear model with parameters $\hat{\theta} = (\hat{a},\hat{b})$

A **loss function** characterizes the cost, error, of fit resulting from a particular choice of model or model parameters.
- Loss quantifies how bad a prediction is for a single observation. 
- If our prediction $\hat{y}$ is close to the actual value $y$, we want low loss.
- If our predition $\hat{y}$ is far from the actual value $y$, we want high loss.

Squared Loss: $L(y,\hat{y})= (y-\hat{y})^2$
Absolute Loss:$L(y,\hat{y})=|y-\hat{y}|$

Average loss (aka empirical risk) for entire data set 
$$
R(\theta) = \frac{1}{n}\sum^{n}_{i=1} L(y_i,\hat{y}_i)
$$
Average loss is a function of the parameter $\theta$ because our data do not change. 
The average loss of a model tells us how well it fits the given data.

$$
R(a,b) = \frac{1}{n}\sum^{n}_{i=1}(y_i-(a+bx_i))^2
$$
**Objective function**: In optimization theory, the function to minimize.
$$
\begin{align}
&\hat{b} =r\frac{\sigma_y}{\sigma_x}\\
&\hat{a} = \overline{y} -\hat{b}\overline{x}\\
& r = \frac{1}{n}\sum^{n}_{i=1}\left(\frac{x_i-\overline{x}}{\sigma_x}\right)\left(\frac{y_i-\overline{y}}{\sigma_y}\right)
\end{align}
$$
Ways to derermine if our model was a good fit to our data
1. Visualize data, compute statistics
2. Performance metrics: Root mean square error
3. Visualization.

## Constant Model, Loss, and Transformations 

**Estimation** is the task of using data to determine model parameters.
**Prediction** is the task of using a model to predict outputs for unseen data.

The **constant model**, also known as a **summary statistic**, summarizes the sample data by always "predicting" the same number i.e., predicting a constant.
It ignores any relationships between variables.
The constant model is also a parametric, statistical model:
$$
\hat{y}= \theta
$$

$$
R(\theta)=\frac{1}{n}\sum^{n}_{i=1}(y_i-\theta)^2
$$
we fit the model by finding the optimal $\hat{\theta}$ that minimizes the MSE.
$$
\hat{\theta}=\textbf{mean}(y)=\overline{y}
$$

Mean Absolute Error (MAE)
$$
R(\theta) = \frac{1}{n}\sum^{n}_{i=1}|y_i-\hat{\theta}|
$$

MSE：
1. **Smooth**: Easy to minimize using numerical methods
2. **Senstive** to outliers.
MAE: 
1. **Piecewise**: at each of the "kinks", it's not differentiable. Harder to minimize.
2. **Robust** to outliers.

Ideal model evaluation steps, in order:
1. Visualize original data,  
    Compute Statistics
2. Performance Metrics  
    For our simple linear least square model,  
    use RMSE (we’ll see more metrics later)
3. Residual Visualization
4. datasets could have similar aggregate statistics but still be wildly different.

**Multiple Linear Regression Model**
$$
\hat{y} =\theta_0+\theta_1x_1+\theta_2x_2+\cdots +\theta_px_p
$$
Parameters are $\theta = (\theta_0,\theta_1,\dots,\theta_p)$

## Ordinary Least Squares 

An expression is "**linear in theta**" if it is a **linear combination** of parameters $\theta = (\theta_0,\theta_1,\dots,\theta_p)$

Matrix:
$$
\hat{\mathbb{Y}} = \mathbb{X}\theta
$$
$\hat{\mathbb{Y}}$ Prediction vector  $\mathbb{X}$ Design matrix   $\theta$ Parameter vector

The least squares estimate $\hat{\theta}$ is the parameter that minimizes the objective function $R(\theta)$
$$
R(\theta) = \frac{1}{n}\lVert\mathbb{Y}-\mathbb{X}\theta\rVert^2_2
$$

Equivalently, this is the $\hat{\theta}$ such that the residual vector $\mathbb{Y} - \mathbb{X}\hat{\theta}$ is orthogonal to $\text{span}(\mathbb{X})$ 
The **normal equation**
$$
\mathbb{X}^T\mathbb{X}\hat{\theta} =\mathbb{X}^T \mathbb{Y}
$$
if $\mathbb{X}^T\mathbb{X}$ is invertible 
$$
\hat{\theta} = (\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T\mathbb{Y}
$$

We define the multiple $\text{R}^2$ value as the proportion of variance or out fitted values(predictions) $\hat{y}$ to our true values $y$
$$
R^2 = \frac{\text{variance of fitted values}}{\text{variance of }y}=\frac{\sigma^2_{\hat{y}}}{\sigma^2_y}
$$
Also called the **correlation of determination**

## Gradient Descent, sklearn 
### sklearn 
`sklearn.learn_model.LinearRegression`

First we create a model. 
```python
model = LinearRegression()
```
Then we "fit" the model, which means computing the parameters that minimize the loss function. The first argument of the fit function should be a matrix (or DataFrame), and the second should be the observation we're trying to predict.
```python
modol.fit(df[["total_bill"]],df["tip"])
```

### Gradient
The key insight is this: If the derivative of the function is negative, that means the function is decreasing, so we should go to the right (i.e. pick a bigger x). If the derivative of the function is positive, that means the function is increasing, so we should go to the left (i.e. pick a smaller x).
$$
x^{(t+1)}= x^{(t)}- \alpha \frac{d}{dx}f(x)
$$

Gradient descent algorithm: nudge $\theta$ in negative gradient direction until $\theta$ converges.
For a model with multiple parameters:
$$
\overset{\rightarrow}{\theta}^{(t+1)} = \overset{\rightarrow}{\theta}^{(t)}-\alpha \nabla_{\overset{\rightarrow}{\theta}}L(\overset{\rightarrow}{\theta},\mathbb{X},\overset{\rightarrow}{y})
$$
$\theta$: Model weights $L$: loss function 
$\alpha$: Learning rate(ours was constant, but other techniques have a $\alpha$ decrease over time)
$y$: True values from training data. 

Impratical in some circumstances:
- Computing the gradient would require computing the loss for a prediction for EVERY data point, then computing the mean loss acorss all several billion.
In mini-batch gradient descent, we only use a subset of the data when computing the gradient.
"Compute gradient on first 10\% of the data. Adjust parameters. Then compute gradient on next 10\% of the data ... Then compute gradient on final 10\% of the data. Adjust parameters"
Each pass is called a training epoch.

f is convex if 
$$
tf(a)+(1-t)f(b)\ge f(ta+(1-t)b)
$$
For a **convex** function f, any local minimum is also a global minimum.

Rather than having to create an entirely new conceptual framework, a better solution is simply to add a new squared feature to our model.

**Feature Engineering** is the process of **transforming** the raw features **into more informative features** that can be used in modeling or EDA tasks.
A **Feature Function** takes our original **d dimensional input** and transforms it into **p demensional input** 
$$
X \in \mathbb{R}^{n \times d}\rightarrow \Phi \in \mathbb{R}^{n \times p}
$$
Designing feature functions is a major part of data science and machine learning.

**One hot encoding**
- Give every category its own feature, with value = 1 if that category applies to that row.
- Can do this using that get_dummies function. Then join with the original table with pd.concat
```python
dummies = pd.get_dummies(data['day'])
data_w_dummies = pd.concat([three_feature_data,dummies],axis = 1)
```

As we increase the complexity of our model:
- Training error decrease.
- Variance increase.
![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUceIeKW-fnW9fxuDLBGRF1lKJ1DNJSIFdfMp4TRW-yWrHCKoJ5VNZvX3PVD8RIaIZ8ChQ1I3_8SDZEr83mbT5kxleUFbmkryn47Vmi8q_iC3TLHxHi4adEwvXo7_xlv6iq-yQGddt12jJNI0zswGmmFiqMfeM_c=nw?key=Dv4AkLzWooAUEz_lWhMScg)
## Cross Validation, Regularization 

### Cross Validation 

The simplest approach for avoiding overfitting is to keep some of our data secret from ourselves. 

**The holdout method**
We train our models on all 25/35 of the available data points. Then we evaluate the model's performance on the remaining 10 data points.
- Data used to used to train is called the **training set**.
- Held out data is often called the "validation set" or "development set" or "dev set". These terms are all synonymous and used by different authors.
```python
from sklearn.utils import shuffle
training_set, dev_set = np.split(shuffle(vehicle_data_sample_35),[25])
```
As we increase the complexity of our model
- Training error decrease.
- Variance increase.
- Typically, error on validation data decreases, the increase.
We pick the model complexity that minimizes validation set error.
![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdxnBw43k6gljmyY_WWQIj1UWZYHDjzLa3PTu2-BnCNlp-N76n-nqtuRWjUwtbRoSVgKUwLERGZggEXErXo9qbEnYTi7ew8GQZVDxjS_Z17a8QCEuJ_OUJDr6aFgbgQP4Q9GtG2f9P5EvQDQdQMvMwTdy62Zifj=nw?key=Dv4AkLzWooAUEz_lWhMScg)
In machine learning, a **hyperparameter** is a value that controls the learning process itself.
We use:
- The training set to select parameters.
- The validation set to select hyperparameters
To determine the quality of a particular hyperparameter
- Train model on **ONLY** the training set. Quality is model's error on ONLY the validation set. 

**K-Fold Cross Validation**
In the k-fold cross-validation approach, we split our data into k equally sized groups.
Given k folds, to determine the quality of a particular hyperparameter.
- Pick a fold, which we'll call the validation fold.Train model on allbut this fold. Compute error on the validation fold.
- Repeat the step above for all k possible choices of validation fold
- Quality is the average of the validation fold errors.

When selecting between models, we want to pick the one that we believe would generallize best on unseen data. Generalization is estimated with a "cross validation score"

Two techniques to compute a “cross validation score”:
- The Holdout Method: Break data into a separate training set and validation set. 
	- Use training set to fit parameters (thetas) for the model.
	- Use validation set to score the model. 
	- Also called “Simple Cross Validation” in some sources.
- K-Fold Cross Validation: Break data into K contiguous non-overlapping “folds”.
	- Perform K rounds of Simple Cross Validation, except:
		- Each fold gets to be the validation set exactly once.
		- The final score of a model is the average validation score across the K trials.

### Test sets 
Our validation set loss is not an unbiased estimator of its performance.

Test sets can be something that we generate ourselves. Or they can be a common data set whose solution is unknown.

```python
diamond_data = shuffle(diamond_data)
# split our 2000 rows into 1500 for training, 300 for validation, 200 for test
diamond_training_data, diamond_validation_data, diamond_test_data = np.split(diamond_data,[1500,1800])
```

Then we use np.split, now providing two numbers instead of one. For example, the code above splits the data into a <font color="#1f497d">Training</font>, <font color="#f79646">Validation</font>, and <font color="#9bbb59">Test set</font>.
- Recall that a <font color="#de7802">validation set</font> is just another name for a <font color="#de7802">development set</font>.
- <font color="#245bdb">Training set</font> used to pick parameters.
- <font color="#de7802">Validation set</font> used to pick hyperparameters (or pick between different models).
- <font color="#9bbb59">Test set</font> used to provide an unbiased MSE at the end.

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfpXIrL-I8fhp9TFougjM9VZXAiM09OeMKJqhmJOoHMvWVSMxHyu_xKDfvqYfj_0Y2j8NiBh6fX8Lrw4l18C5dLlVapE97t4JaDC2GqQ88m6NIupuAfFT6er0WCoCP27yO1evpDfFRqmQNhM2RNa7I0Dovj-M0=nw?key=Dv4AkLzWooAUEz_lWhMScg)


### Regularization(Ridge) 
Constaining our model's parameter to a ball around the origin is called L2 Regularization.
- The smaller the ball, the simpler the model.

We can run least squares with an L2 regularization term using the "Ridge" class.
```python
form sklearn.linear_model import ridge
ridge_model = Ridge(alpha = 10000)
ridge_model.fit(vehicle_data_with_squared_features, vehicle_data["mpg"])
```
$$
\frac{1}{n}\sum^{n}_{i=1}(y_i -(\theta_0+\theta_1\phi_{i,l}+\cdots+\theta_d\phi_{i,d}))^2+\alpha\sum^{d}_{j=1}\theta^2_j
$$

## SQL 

Every column in a SQL table has three properties: ColName, Type, and zero or more Constraints.
**Some examples of SQL types**:
- INT: Integers
- REAL: Real numbers 
- TEXT: Strings of text 
- BLOB: Arbitrary data 
- DATATIME: A data and time 
**Some examples of constraints**: 
- CHECK: Data cannot be inserted which violates the given check constaint
- PRIMARY KET: Specifies that this key is used to uniquely identify rows in the table.
- NOT NULL: Null data cannot be inserted for this column 
- DEFAULT: Provides a value to use if user does not specify on insertion 

We can also SELECT only a subset of the columns.
The AS keyword lets us rename columns during the selection process.
To select only some rows of a table, we can use the WHERE keyword.
The OR keyword lets us form a disjunctive condition.
The ORDER BY keyword lets you sorts a Table.
The LIMIT keyword lets yor retrieve N rows.
The OFFSET keyword lets you tell SQL to see later rows when limiting. 

To filter:
- Rows, use WHERE.
- Groups, use HAVING 

The LIKE operator tests whether a string matches a pattern
The CAST keyword converts a table column to another type. 

## PCA

Suppose we have a dataset of 
- **n** observations(datapoints)
- **d** attributes(features)

**Intrinsic Dimension** of a dataset is the **minimal set of dimensions** needed to approximately represent the data

**Dimension of the column space** of A is the rank of matrix A.

In general, we want the projection that is the best approximation for the original data.
In other words, we want the projection that capture **the most variance** of the original data.

Find a linear transformation that creates a low-dimension representation which captures as much of the original data's **total variance** as possible.

1. Data matrix preprocessing: Oftentimes you will center the data matrix by subtracting the mean of each attribute column.
2. To find the i-th principal component $v_i$:
	- $v$ is a unit vector that linearly combines the attributes.
	- $v$ gives a one-dimensional projection of the data.
	- $v$ is chosen to minimize the sum of squared distances between each point and its projection onto $v$.
	- Choose $v$ such that it is orthogonal to all previous principal components.

PCA with SVD
$$
\begin{align}
&X = USV^T\\
&XV = U S
\end{align}
$$
PC1 = (1st left singular vector)\*(1st singular value)
PC2 = (1st left singular vector)\*(2nd sigular value)

How to Obtain Principal Components 
1. Preprocess $X$, usually centering and sometimes scaling.
2. Compute SVD on preprocessed $X$: $X = USV^T$
3. Take the firts $k$ columns of $XV$ or $US$. These are the first $k$ principal components
4. If we wanted to get rank-$k$ approximation of $X$

Formally, the $i$th singular value tells us the component score.

## Logistic Regression 

Logistic Regression is what we call a generalized linear model.
- Non-linear transformation of a linear model.
- So parameters are still a linear combination of: $x^T\theta$ 

The logistic function
$$
\sigma(t)=\frac{1}{1+e^{-t}}
$$
This is a type of sigmoid, a class of functions that share certain properties.

Logistic Regression Model assumptions: 
- Fit the "S" curve as best as possible 
- The curve models probability $P(Y =1 |x)$
- Assume log-odds is a linear combination of $x$ and $\theta$

The logistic regression model:
$$
\hat{P}_\theta(Y=1|x) = \sigma(x^T\theta)
$$

3 pitfalls of squared loss
1. Non-convex 
2. Bounded
3. Conceptually questionable

The **cross-entopy loss** is defined as 
$$
-(y\log(p)+(1-y)\log(1-p))
$$

The optimization problem is therefore to find the estimate $\hat{\theta}$ that minimizes $R(\theta)$
$$
\hat{\theta} = \underset{\theta}{\text{argmin}}-\frac{1}{n}\sum^{n}_{i=1}(y_i\log(\sigma(X^T_i\theta))+(1-y_i)\log(1-\sigma(X^T_i\theta)))
$$

If binary data are independent with different probability $p_i$, then the likelihood of the data is 
$$
\prod^n_{i=1}p^{y_i}_i(1-p_i)^{(1-y_i)}
$$
So our maximum likelihood problem is to find $\hat{p}_1,\hat{p}_2,\dots,\hat{p}_n$ that maximize $\prod^n_{i=1}p^{y_i}_i(1-p_i)^{(1-y_i)}$
And that is 
$$
-\frac{1}{n}\sum^{n}_{i=1}(y_i\log(\sigma(X^T_i\theta))+(1-y_i)\log(1-\sigma(X^T_i\theta)))
$$

Minimizing cross-entopy loss is equivalent to maximizing the likelihood of the training data

Objective Function: Average Cross-Entropy Loss + regularization

A classification dataset is said to be linearly separable if there exists a hyperplane among input features x that separates the two classes y

To avoid large weights, use regularization.
$$
\underset{\theta}{\text{argmin}} - \frac{1}{n}\sum^{n}_{i=1}(y_i\log(\sigma(X^T_i\theta))+(1-y_i)\log(1-\sigma(X^T_i\theta))) + \lambda \sum^{d}_{j=1}\theta_j^2
$$

The most basic evaluation metric for a classifer is **accuracy**
$$
\text{accuracy}=\frac{\text{\# of classified correctly}}{\text{\# points total}}
$$
While widely used, the accuracy metric is not so meaningful when dealing with class imbalance in a dataset.

The Confusion Matrix 
- **True positives(TP)** and **True negatives(TN)** are when we correctly classify an observation as being positive or negative, respectively.
- **False positives(FP)** are “false alarms”:  we predicted 1, but the true class was 0.
- **False negatives(FN)** are “failed detections”:  we predicted 0, but the true class was 1.

**Precision** and **recall** are two commonly used metrics that, measure performance even in the presence of class imbalance.
$$
\text{precision} = \frac{TP}{TP+FP}
$$
- How accurate is our classifier when it is positive
- Penalizes false positives

$$
\text{recall} = \frac{TP}{TP+FN}
$$
- How sensitive is our classifier to positives
- Penalizes false negatives.

Classification Threshold:
$$
\hat{y} = \text{classify}(x)=\left\{\begin{align}
&1 \quad \hat{P}_\theta(Y=1|x)\ge T\\
&0 \quad \text{otherwise}
\end{align}\right.
$$

The choice of threshold T impacts our classification performance
- High T: Most predictions are 0. Lots of false negatives.
- Low T: Most predictions are 1. Lots of false positive.

Two More Metrics 
True Positive Rate(TPR)
$$
\text{TPR} = \frac{\text{TP}}{\text{TP+FN}}
$$
False Positive Rate(FPR)
$$
\text{FPR} = \frac{\text{FP}}{\text{FP+TN}}
$$
As we increase T, both TPR and FPR decrease.
- A decrease TPR is bad (detecting fewer positives).
- A decrease FPR is good (fewer false positives).

The "perfect" classifier is the one that has a TPR of 1, and FPR of 0

We can compute the area under curve(AUC) of our model.
Best possible AUC = 1. Terrible AUC = 0.5 
Your model's AUC: somewhere between 0.5 and 1

Muticlass Classification
```python
logistic_regression_model = LogisticRegression(multi_class = 'over')
logistic_regression_model = logistic_regression_model.fit(iris_data[["petal_length","petal_width"]],iris_data["species"])
```
## Decision Trees 

A Decision Tree is an alternate way to classify data. It is simply a tree of questions that must be answered in sequence to yield a predicted classification.

```python
from sklearn import tree
decision_tree_model = tree.DecisionTreeClassifier(criterion = 'entropy')
decision_tree_model = decision_tree_model.fit(iris_data[["petal_length","petal_width"]],iris_data["species"])
```

Traditional decision tree generation algorithm:
- All of the data starts in the root node.
- Repeat until every node is either pure or unsplittable:
	- Pick the best feature $x$ and best split value $\beta$ such that the $\Delta \text{WS}$ is maximized
	- Split data into two nodes, one where $x<\beta$, and one where $x\ge \beta$

Entropy S of a node(in bits) as 
$$
S = -\sum^{}_{c}p_c\log_2p_c
$$
where $p_c$ is the proportion of data points in a node with label C

Define the weighted entropy of a node as its entropy scaled by the fraction of the samples in that node.
The weighted entropy always decreases as we move down the tree.

A node that has only one samples from one class is called a "**pure**" node. A node that has overlapping data points from different classes and thus cannot be split is called "**unsplittable**"

A "fully grown" decision tree built our algorithm runs the risks of overfitting.
- Regularization doesn't make sense in the decision tree context. There's no weights to be penalized.

**Approaches to avoid overfitting**
1. Set one or more special rules to prevent growth.
2. Let tree fully grow, then cut off less useful branches of the tree.

There's a completely different idea called a "random forest" that is more popular and IMO more beautiful.

Fully-grown decision trees will almost always overfit data
- Low model bias, high model variance
- Small changes in dataset will result in very different decision tree.

Random Forest Idea: Build many decision trees and have them vote.

Bagging: Short for Bootstrap AGGregatING 
- Generate bootstrap resamples of training data.
- FIt one model for each resample.
- Final model = average predictions of each small model.

- Bagging often isn't enough to reduce model variance.
	- Decision trees often look very similar to each other.
	- Ensemble will still have low bias and high model variance.	

Bootstrap training data T times. For each resample, fit a decision tree by doing the following:
	- Start with data in one node. Until all nodes pure or unsplitable:
		- Pick an impure node.
		- Pick a random subset of $m$ features. Pick the best feature $x$ and split value $\beta$ such that the loss of the resulting split is minimized
		- Split data into two nodes, one where $x<\beta$ and one where $x\ge beta$
	- To predict, ask the T decision trees for their predictions and take majority vote.
This approach has two hyperparameters T and m.

## Clustering 
In "Unsupervised Learning":
- Goal is to identify patterns in unlabeled data
	- We do not have input/output pairs
Goal of clustering: Assign each point to a cluster.

### K-Means Clustering 
Most popular clustering approach .
- Pick an arbitrary $k$, and randomly place k "centers", each a different color.
- Repeat until convergence:
	- Color points according to the closest center.
	- Move center for each color to center of points with that color.

To evaluate different clustering results, we need a loss function.
- **Inertia**: Sum of squared distances from each data point to its center.
- **Distortion**: Weighted sum of squared distances from each data point to its center.

Optimizing Inertia algorithm
- For all possible $k^n$ colorings:
	- Compute the k centers for that coloring.
	- Compute the inertia for the k centers.
		- If current inertia is better than best known, write down the current centers and coloring and call that the new best known.

### Agglomerative clustering 
Basic idea:
- Every data point starts out as its own cluster.
- Join clusters with neighbors until we have only K clusters left.

Agglomerative clustering is one form of "hierarchical clustering"
- Can keep track of when two cluster got merged
	- Each cluster is a tree.
- Can visualize merging hierachy, resulting in a "dendrogram"

### Picking K 
The algorithm we've discussed today require us to pick a K before we start.

For K-Means, one approach is to plot inertia versus many different K values.
- Pick the K in the "elbow", where we get diminishing returns afterwards.
- Big complicated data often lacks an elbow.

To evaluate how "well clustered" a specific data point is, we can use the "sihouette score".
- High score: Near the other points in its X's cluster.
- Low score: Far from the other points in its cluster.