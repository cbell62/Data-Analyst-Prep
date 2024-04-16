<a name="readme-top"></a>
<div align="center">
<h3>Data Analyst Prep</h3>
</div>

<details open="true">
  <summary><strong> :page_with_curl: Table of Contents</strong></summary>
  <ol>
      <li><a href="#approaching-a-new-project">Approaching A New Project Steps</a></li>
      <li><a href="#data-cleansing">Data Cleansing</a></li>
      <ul>
         <li><a href="#deduplication">Data Deduplication</a></li>
         <li><a href="#handling-missing-values">Handling Missing Values</a></li>
         <li><a href="#validating-accuracy-with-descriptive-statstics">Validating Accuracy with Descriptive Stats</a></li>
         <li><a href="#converting-text-to-lowercase-or-removing-unnecessary-characters">Converting text to lower case, removing unnecessary characters</a></li>
         <li><a href="#transforming-data">Transforming Data</a></li>
         <li><a href="#standardizing-data">Standardizing Data</a></li>
         <li><a href="#data-validation">Data Validation</a></li>
         <li><a href="#outlier-detection">Outlier Detection</a></li>
         <li><a href="#best-practices-for-data-cleansing"> Data Cleansing Best Practices</a></li>
         <li><a href="#preferred-tools"> Preferred Tools for Data Analysis </a></li>
      </ul>
      <li><a href="#data-mining-versus-data-analysis">Data Mining versus Data Analysis</a></li>
      <li><a href="#well-developed-data-model-versus-not-well-developed-data-model">Well Developed Data Model versus Not a Well Developed Model</a></li>
      <li><a href="#typical-problems-encountered-during-analysis">Typical Problems Encountered During Analysis</a></li>
      <li><a href="#retraining-model-when-and-how-often"> When and How often to Retrain a Model </a></li>
      <li><a href="#resolving-multisource-problems-using-data-integration"> Resolving Multisource Problems with Data Integration </a></li>
      <li><a href="#imputation-and-some-methods"> Imputation and Some Methods </a></li>
      <li><a href="#collaborative-filtering-with-example"> Collaborative Filtering. What it is and an Example </a></li>
      <li><a href="#map-reduce-and-its-purpose"> Purpose of Map-Reduce </a></li>
      <li><a href="#exploratory-data-analysis-and-why-it-is-important"> EDA and Why it is Important </a></li>
      <li><a href="#common-eda-techniques"> Common EDA techniques </a></li>
  </ol>
</details>

### Approaching A New Project
1. Understand the Project Goals: The first step is to understand the objectives and goals of the project. What are the key questions that the project aims to answer? What business problems are we trying to solve?
2. Identify Stakeholders: Identify who will be directly affected by the project and who will be making decisions. These could be internal team members, clients, or other departments within the organization.
3. Data Collection: Identify the necessary data sources. This could be internal databases, third-party data, public data sets, etc. Ensure you have the necessary permissions and access rights to these data sources.
4. Data Cleaning: Clean the collected data to ensure it’s accurate and reliable. This includes handling missing values, removing duplicates, and checking for inconsistencies.
5. Exploratory Data Analysis (EDA): Conduct an initial analysis to understand the patterns, anomalies, or relationships in the data. This can help in forming hypotheses and further analysis.
6. Data Modeling: Based on the project goals, select appropriate statistical models or machine learning algorithms to analyze the data. This could involve predictive modeling, classification, clustering, etc.
7. Interpret Results: Translate the results from your analysis into actionable business insights. Make sure these are aligned with the project goals.
8. Communicate Findings: Present your findings to the stakeholders in a clear and understandable manner. Use visualizations to help communicate complex data insights.
9. Implement Solutions: Depending on the project, this could involve developing a new data-driven strategy, building a predictive model, or implementing a new data management process.
10. Review and Iterate: After implementation, review the project outcomes against the original goals. Use this feedback to refine and optimize the process.


## Data Cleansing

### Deduplication

1. Pandas in Python - removing duplcate rows
```
import pandas as pd

df = df.drop_duplicates()

```

2. SQL: if data in sql database, use distinct to return unique rows

```
SELECT DISTINCT column1, column2, ..., columnN
From table_name;
```

3. Excel: remove duplicates feature under the data tab

4. R: duplicated or unique functions to remove duplicates from a data frame
```
df = df[!duplicated(df, )]
```

### Handling Missing Values

1. Pandas in Python
```
import pandas as pd

#identify missing values
missing_values = df.isnull()

#remove missing values
df_no_missing = df.dropna()

#replace missing values
df_filled = df.fillna(value)

```

2. SQL - Use ISNULL or IS NOT NULL to identify missing values. Use COALESCE() function or CASE statement to replace missing values
```
Identify missing values
SELECT colummn_name 
From table_name
Where column_name IS NULL;

-- Replace missing values
UPDATE table_name
SET column_name = COALESCE (column_name, value);
```
3. Excel - Use the ISBLANK function to identify missing values. Use go to special --> blanks feature to seleect all blank cells and then replace them in order to handle missing values

### Validating Accuracy with Descriptive Statstics
``` 
import pandas as pd

df.describe()
```

### Converting text to lowercase or removing unnecessary characters
1. Python
```
df['column'] = df['column'].str.lower() 
df['column'] = df['column'].str.replace('[^\\w\s]', '')
```
2. SQL 
```
UPDATE your_table
SET column - LOWER(column); -- convert text to lowercase

UPDATE your_table
SET column = REPLACE(column, 'old_String', 'new_String'); -- replace old string with new string

UPDATE your_table
SET column = TRIM (column); -- remove leading and trailing spaces

```
### Transforming Data
1. Python
``` 
# Converting categirial data to numerical data
df['column'] = pd.Categorial(df['column'])
df['column'] df['column'].cat.codes
```
2. SQL - use case statenebt to convert categorial data to numerical
```
UPDATE your_table
SET column = CASE
    WHEN column = 'category1' THEN 1
    When column = 'category2' Then 2
    Else 3
END;
```
### Standardizing Data

Z-Score Normalization: transforms data to have mean of 0 and standard deviation of 1. 

1. Python - Implemented using standardscaler class in sklearn.preprocessing

```
from sklearn.preprocessing import StandardScaler
import numpy as np

#assuming x is your data
scaler = StandardScaler
X_standardized = scaler.fit_transform(X)

#assuming you are using a dataset
df['column'] = scaler.fit_transform(df[['column']])
```

2. SQL - In SQL can be achieved using formula for z score normalization ```z = x- u/ o``` where x is each value in column, u is mean of column, and o is standard deviation of column

```
UPDATE your_table
SET column = (column - (SELECT AVG(column) FROM your_table)) /
             (SELECT STDDEV(column) FROM your_table);
```

### Data Validation
1. Python - This involves checking if the data meets certain criteria, rules, or standards. In Python, you can use the pandas library to validate your data.

```
import pandas as pd

# Assuming df is your DataFrame and 'column' is the column you want to validate
if df['column'].dtype != 'int64':
    print("Invalid data type in column. Expected int64.")
```

2. SQL - This involves checking if the data meets certain criteria, rules, or standards. In SQL, you can use the DATA_TYPE information from the INFORMATION_SCHEMA.COLUMNS table to validate your data.
```
SELECT COLUMN_NAME, DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'your_table' AND COLUMN_NAME = 'column';
```
### Outlier Detection
1. Python - Outliers are data points that are significantly different from other observations. They can be detected using various methods. One common method is the Z-score method.
```
from scipy import stats
import numpy as np

# Assuming df is your DataFrame and 'column' is the column you want to check for outliers
z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]  # Change '3' to a different value to adjust the threshold

```

In the above code, the zscore function is used to calculate the Z-score of each value in the column, which measures how many standard deviations a point is from the mean. A common practice is to consider values with a Z-score greater than 3 as outliers, but this threshold can be adjusted depending on the specific context.

2. SQL -  does not directly support the calculation of Z-scores. You would need to calculate the mean and standard deviation first, and then use these to calculate the Z-scores.
```
-- Calculate the mean and standard deviation
SELECT AVG(column) AS mean, STDDEV(column) AS stddev
FROM your_table;

-- Calculate the Z-scores
SELECT column, (column - mean) / stddev AS z_score
FROM your_table;

```

In the above code, the AVG and STDDEV functions are used to calculate the mean and standard deviation of the column. Then, these values are used to calculate the Z-score of each value in the column, which measures how many standard deviations a point is from the mean. A common practice is to consider values with a Z-score greater than 3 as outliers, but this threshold can be adjusted depending on the specific context.
### Best Practices for Data Cleansing
- **Identify duplicates**: This involves finding and removing duplicate records from the dataset. In Python, you can use the `duplicated()` and `drop_duplicates()` functions from pandas.
    ```python
    import pandas as pd
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                       'C': np.random.randn(8),
                       'D': np.random.randn(8)})
    print(df.duplicated())
    df = df.drop_duplicates()
    ```
    - **Deal with missing values**: Depending on the situation, you might fill the missing values with a specific value, the mean or median of the column, or you can drop the rows with missing values.
    ```python
    df.fillna(0)
    df.dropna()
    ```
    - **Data type conversions**: Ensure that your data types are correct. For example, a numerical value could be read as a string. You can use the `astype()` function to convert data types.
    ```python
    df['column'] = df['column'].astype('int')
    ```
    - **Scaling and normalization**: When dealing with features that are on different scales, you might want to scale the features to a range or standardize them to have a zero mean and unit variance.
    ```python
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = MinMaxScaler()
    df['column'] = scaler.fit_transform(df[['column']])
    ```
### Preferred Tools 
Tools preferred for data analysis often depend on the specific requirements of the project. However, some commonly used tools include:
    - **Python**: It's a versatile language with libraries like Pandas for data manipulation, Matplotlib and Seaborn for data visualization, and Scikit-learn for machine learning.
    - **R**: This is another language that's very popular in statistics and data analysis.
    - **SQL**: Used for querying databases.
    - **Excel**: Great for quick data analysis and manipulation.
    - **Tableau**: A powerful tool for creating interactive data visualizations.
## Data Mining versus Data Analysis

1. Data Analysis:
   Definition: Data analysis involves interpreting data to find trends and patterns.
- Approaches:
    Descriptive Analytics: Focuses on understanding past events by generating reports or building dashboards.
    Predictive Analytics: Uses historical data to build models for future predictions (e.g., customer behavior, employee attrition).
    Prescriptive Analytics: Not only predicts future outcomes but also recommends actions to achieve desired results.
    Example: Analyzing sales data to identify demographics more likely to purchase a product and targeting marketing efforts accordingly.
2. Data Mining:
  Definition: Data mining is a specific type of data analysis that focuses on finding hidden patterns and relationships in large datasets.
  Objective: Extract valuable information from data.
  Applications:
    Fraud Detection: Identifying unusual patterns in financial transactions.
    Marketing: Finding groups of customers with similar characteristics.
  Skills Needed: Requires understanding of statistics and computer programming.
  Example: Extracting insights from customer purchase history, such as popular products among specific demographics.

In summary:

Data Analysis interprets data.
Data Mining extracts essential information from datasets.
## Well-Developed Data Model versus Not Well-Developed Data Model
A well-developed data model differs from a poorly developed one in several key aspects:

1. **Clarity and Simplicity**: A well-developed data model is easy to understand and interpret. It avoids unnecessary complexity and focuses on representing the essential aspects of the data.

2. **Consistency**: The model should be consistent in its representation of data. This means that similar types of data should be modeled in the same way, and the rules and constraints applied to the data should be uniformly enforced.

3. **Accuracy**: The model should accurately represent the real-world entities and relationships it is intended to capture. It should reflect the true nature of the data, not just the way the data is currently being used.

4. **Flexibility**: A good data model can accommodate changes in requirements or business rules without requiring a major redesign. It should be designed with future growth and evolution in mind.

5. **Efficiency**: The model should be designed in a way that supports efficient data operations, such as queries and updates. This often involves trade-offs between normalization (to eliminate redundancy) and denormalization (to optimize performance).

6. **Integrity**: The model should enforce data integrity through constraints and relationships. This includes things like referential integrity (ensuring that foreign key values always point to existing rows) and domain integrity (ensuring that data values fall within specified ranges).

7. **Comprehensiveness**: A well-developed data model covers all necessary aspects of the information domain. It doesn't leave out relevant data or relationships.

Remember, the goal of a data model is to organize, define, and standardize the data elements in a system to support the system's requirements in the best possible way. A well-developed data model serves as a solid foundation for your database and applications, ensuring data consistency, accuracy, and usability.

## Typical Problems Encountered During Analysis
Data analysts can encounter a variety of challenges during their analysis. Here are some common ones:

1. **Data Quality Issues**: This includes missing data, inconsistent data, outliers, and errors in the data. These issues can significantly impact the accuracy of the analysis.

2. **Large Data Volumes**: Handling large amounts of data can be challenging due to storage limitations and performance issues. It can also make data cleaning and preprocessing more difficult.

3. **Complexity of Data**: Data can come from various sources and in different formats. Integrating and making sense of all this data can be complex and time-consuming.

4. **Lack of Domain Knowledge**: Without a good understanding of the domain, it can be difficult to interpret the data correctly and make meaningful insights.

5. **Changing Requirements**: Business requirements can change over time, which might require changes in the analysis. This can be challenging if the data model isn't flexible.

6. **Data Privacy and Security**: Ensuring the privacy and security of sensitive data is a major concern. Analysts need to comply with regulations and ethical guidelines when handling such data.

7. **Communication of Results**: Conveying the results of an analysis in a clear and understandable way can be challenging, especially to non-technical stakeholders.

Remember, each of these challenges can be addressed with the right tools, techniques, and approaches. For example, data quality issues can be mitigated with robust data cleaning processes, while the challenge of large data volumes can be addressed with big data technologies and cloud computing solutions.

## Retraining Model: When and How Often?
Machine learning models should be retrained under the following circumstances:

1. **Data Drift**: This occurs when the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This could be due to a variety of factors, including a change in user behavior, a new product launch, or macroeconomic factors.

2. **Concept Drift**: This is when the relationship between the input data and the output prediction changes over time. For example, a model trained to predict stock prices may need to be retrained as market conditions change.

3. **Availability of New Data**: If new data becomes available that was not available during the initial training, it can be beneficial to retrain the model. This new data could lead to improved model performance.

4. **Performance Degradation**: If the model's performance on key metrics (like accuracy, precision, recall, etc.) starts to decline, it may be time to retrain the model.

As for how often a model should be retrained, it depends on the specific use case and the factors mentioned above. Some models may need to be retrained daily, while others might only need to be retrained monthly or even yearly. It's important to monitor model performance regularly to determine when retraining might be necessary.

## Resolving Multisource Problems Using Data Integration

Regarding multisource problems, these are typically resolved through a process called data integration. This involves combining data from different sources and providing users with a unified view of these data. This process can involve several steps, including:

1. **Data Cleaning**: This step involves identifying and correcting errors in the data, handling missing values, and resolving inconsistencies.

2. **Schema Integration**: If the data sources have different schemas, they need to be integrated into a unified schema.

3. **Data Transformation**: This involves converting data into a suitable format for analysis. This could involve normalizing data, aggregating data, or performing other transformations.

4. **Data Deduplication**: This step involves identifying and removing duplicate records from the data.

5. **Data Fusion**: This is the process of integrating multiple data sources into a single, consistent data set. This can involve resolving conflicts between data sources and deciding how to handle discrepancies.

Remember, dealing with multisource data can be complex, but with the right strategies and tools, it's possible to turn this challenge into an opportunity for more comprehensive analysis.

## Imputation and Some Methods
Imputation is a statistical technique used to handle missing data in datasets. The goal of imputation is to produce a complete dataset that can be used for further analysis. Here are a few different imputation techniques:

1. **Mean/Median/Mode Imputation**: This method involves replacing the missing values with the mean, median, or mode of the available data. It’s simple to implement but can lead to an underestimation of the variance in the data.

```
from sklearn.impute import SimpleImputer
import numpy as np

# Assuming `X` is your data with missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)
```
2. **Constant Imputation**: This method involves replacing the missing values with a constant value. This is typically used when the data is categorical.

```
from sklearn.impute import SimpleImputer

# Assuming `X` is your data with missing values
imputer = SimpleImputer(strategy="constant", fill_value="missing")
X_imputed = imputer.fit_transform(X)
```

3. **K-Nearest Neighbors (KNN) Imputation**: This method involves replacing the missing values with the values from ‘K’ similar instances. It’s more accurate than mean/median imputation but is computationally expensive.
```
from sklearn.impute import KNNImputer

# Assuming `X` is your data with missing values
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)
```
4. **Multiple Imputation by Chained Equations (MICE)**: This is a more sophisticated technique that performs multiple regressions over random sample sets, then averages the data to fill the missing values.

```
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Assuming `X` is your data with missing values
imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X)
```

Remember, the choice of imputation technique depends on the nature of the data and the specific use case. It’s always a good idea to experiment with different methods and choose the one that results in the best model performance. Also, it’s important to note that all imputation methods introduce some level of bias into the data, so it’s crucial to take this into account when interpreting the results of your analysis.

## Collaborative Filtering with Example
**Collaborative filtering** is a technique used by recommender systems to make predictions about the interests of a user by collecting preferences from many users. 
The underlying assumption is that if a user A has the same opinion as a user B on an issue, A is more likely to have B's opinion on a different issue.

Here's a simple example of user-based collaborative filtering using Python and the `scikit-learn` library:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Let's assume we have 5 users and each user has rated 5 different items (like movies, products, etc)
# Ratings are from 1 to 5. A zero indicates that the user hasn't rated that item.
user_item_matrix = np.array([
    [3, 4, 3, 0, 2],
    [3, 0, 4, 3, 1],
    [4, 2, 3, 3, 2],
    [3, 3, 1, 5, 2],
    [2, 3, 0, 3, 2]
])

# Compute the similarity matrix
similarity_matrix = cosine_similarity(user_item_matrix)

# Let's say we want to recommend a new item to the first user
# We'll predict his rating based on the ratings of other users and their similarities to the first user
user_similarity = similarity_matrix[0] # 0 for 1st user 1 for second user and so forth
user_ratings = user_item_matrix[:, 3]  # This is the item we want to recommend
weighted_ratings = user_ratings * user_similarity
predicted_rating = np.sum(weighted_ratings) / np.sum(user_similarity)

print(f"The predicted rating for the first user is: {predicted_rating}")
```

In this example, we're predicting the rating of an item for the first user based on the ratings of that item by other users and their similarity to the first user.
 The similarity is computed using the cosine similarity metric, which is a common choice in collaborative filtering.

Please note that this is a very simplified example. In practice, collaborative filtering can be much more complex and may involve techniques like matrix factorization, handling of sparse data,
and dealing with the "cold start" problem (i.e., making recommendations for new users or items).

## map-reduce and its purpose
**MapReduce** is a programming model and software framework for processing large datasets. It's particularly useful for distributed computing on big data using a cluster of computers (nodes). Here's why it's used:

1. **Handling Big Data**: MapReduce is designed to process datasets that are too large to fit on one single machine. It can handle petabytes of data by splitting them into smaller chunks¹².

2. **Distributed Processing**: MapReduce allows for the concurrent processing of data. It divides the data into independent chunks which are processed by the map tasks in a completely parallel manner¹²⁴.

3. **Fault Tolerance**: MapReduce is designed to handle failures. If a node goes down during computation, the system reroutes the task to another node to avoid a complete failure².

4. **Scalability**: As data grows, you can add more machines to your cluster to handle that data. MapReduce will take care of distributing the data and the computations across the machines².

5. **Simplicity**: Developers only need to focus on writing the Map and Reduce functions for their specific task without worrying about the details of data distribution, parallelization, and fault tolerance².

For example, consider a scenario where you have a large text file and you want to count the frequency of each word. You could use MapReduce to solve this problem in two stages:

- **Map Stage**: Each mapper takes a portion of the text file and outputs a key-value pair for each word in its portion, where the key is the word and the value is 1.

- **Reduce Stage**: Each reducer takes the output from the mappers, aggregates the values for each word, and outputs a final count for each word.

This is a simple example, but MapReduce can be used for much more complex tasks involving large datasets. It's a core component of the Hadoop framework and is used extensively in big data analytics¹²⁴.

Let's consider a simple example to illustrate the purpose of MapReduce: **word count**. This is a common example used to demonstrate the concept of MapReduce. 
The task is to count the number of occurrences of each word in a large dataset of documents.

Here's how it would work:

1. **Map Phase**: In the map phase, the input data (documents) are divided into smaller sub-datasets. Each of these sub-datasets is processed by a map function. The role of the map function is to process a sub-dataset and generate a set of intermediate key-value pairs. For the word count problem, the map function processes a document and produces a key-value pair for each word in the document. The key is the word, and the value is 1, representing one occurrence of the word.

```python
def map(document):
    words = document.split()
    for word in words:
        emit(word, 1)
```

2. **Shuffle and Sort Phase**: After the map phase, the MapReduce framework performs a shuffle and sort operation. All the key-value pairs with the same key are grouped together.

3. **Reduce Phase**: In the reduce phase, the reduce function processes each group of values that share the same key. It combines the values to form a single output value. In the word count problem, the reduce function sums up the counts for each word and emits a key-value pair with the word and its total count.

```python
def reduce(word, counts):
    total_count = sum(counts)
    emit(word, total_count)
```

This MapReduce operation can be performed in parallel on a cluster of computers, which makes it highly scalable for large datasets. The map functions can process different documents on different computers, and the reduce functions can process different words on different computers. This is the key strength of MapReduce: it can turn a big data problem into many small data problems that can be solved in parallel.

## Correlogram Analysis and How it is Used
A **correlogram**, also known as an Auto Correlation Function (ACF) plot, is a chart of correlation statistics¹². It's a visual way to show serial correlation in data that changes over time, i.e., time series data². Serial correlation, also called autocorrelation, is where an error at one point in time travels to a subsequent point in time².

Correlograms are used to assess randomness and identify simple patterns in your data by quickly identifying variables that are strongly correlated with one another⁶. They can give you a good idea of whether or not pairs of data show autocorrelation². However, they cannot be used for measuring how large that autocorrelation is².

In the analysis of data, if random, autocorrelations should be near zero for any and all time-lag separations. If non-random, then one or more of the autocorrelations will be significantly non-zero¹. The correlogram is a commonly used tool for checking randomness in a data set¹. It's used as a tool to check randomness in a data set which is done by computing auto-correlations for data values at different time lags⁴.

In addition, correlograms are used in the model identification stage for Box–Jenkins autoregressive moving average time series models¹. They can help provide answers to the following questions¹:
- Are the data random?
- Is an observation related to an adjacent observation?
- Is an observation related to an observation twice-removed? (etc.)
- Is the observed time series white noise?
- Is the observed time series sinusoidal?
- Is the observed time series autoregressive?
- What is an appropriate model for the observed time series?
- Is the model valid and sufficient?
- Is the formula valid?

The correlogram analysis is a key tool to explore the inter-dependency of the observation values; it can also be used as a tool to identify the model and estimate the orders of its components⁷..

## Hash Table Collison: What it is and How to Avoid
Hash table collisions occur when two different keys hash to the same index in the hash table. There are several strategies to handle these collisions:

1. **Separate Chaining**: In separate chaining, each cell in the hash table points to a linked list of records that have the same hash function value. When a collision occurs, the record is simply added to the end of the list.

```python
def insert(hash_table, key, value):
    hash_key = hash(key) % len(hash_table)
    key_exists = False
    bucket = hash_table[hash_key]    
    for i, kv in enumerate(bucket):
        k, v = kv
        if key == k:
            key_exists = True 
            break
    if key_exists:
        bucket[i] = ((key, value))
    else:
        bucket.append((key, value))
```

2. **Open Addressing (Linear Probing)**: In open addressing, when a collision occurs, we look for the next available slot or address in the hash table. In linear probing, we linearly probe for the next empty cell.

```python
def insert(hash_table, key, value):
    hash_key = hash(key) % len(hash_table)
    while hash_table[hash_key] is not None:
        hash_key = (hash_key + 1) % len(hash_table)
    hash_table[hash_key] = value
```

3. **Double Hashing**: Double hashing uses a secondary hash function when collisions occur. The main idea is to calculate an offset and probe the table at that offset.

```python
def insert(hash_table, key, value):
    hash_key = hash(key) % len(hash_table)
    if hash_table[hash_key] is None:
        hash_table[hash_key] = value
    else:
        new_hash_key = (hash_key + (1 + (hash(key) % (len(hash_table) - 1)))) % len(hash_table)
        while hash_table[new_hash_key] is not None:
            new_hash_key = (new_hash_key + (1 + (hash(key) % (len(hash_table) - 1)))) % len(hash_table)
        hash_table[new_hash_key] = value
```

Remember, the choice of collision resolution technique can be critical for the performance of the hash table, and the best method depends on the specifics of the use case. It's also important to note that avoiding collisions entirely is impossible, but a good hash function will minimize them.

###
## Exploratory Data Analysis and Why it is Important?
Exploratory Data Analysis (EDA) is a crucial step in the data analysis pipeline. Here are some reasons why it's important:

1. **Understanding the Data**: EDA helps to understand the main characteristics of the data, its structure, and its distribution. It provides a better understanding of the variables and the relationships between them.

2. **Identifying Patterns and Relationships**: EDA can help identify patterns, relationships, or correlations between variables in the dataset. These insights could be useful for predicting or explaining the behavior of the data.

3. **Detecting Anomalies**: EDA can help detect anomalies and outliers in the data. These could be errors or interesting data points that could lead to further investigation.

4. **Assumptions Testing**: EDA is used to test assumptions or hypotheses for further statistical analysis. It helps determine if the data meets the assumptions required by certain statistical tests.

5. **Preparing for Modeling**: EDA can inform the preprocessing steps required for machine learning models, such as feature selection, feature engineering, and setting up a validation strategy.

6. **Communicating Results**: Visual methods used in EDA are helpful for summarizing the data and communicating the results to others.

In summary, EDA is about making sure that the data you have collected makes sense and is ready for more advanced analysis. It can save a lot of time and reveal insights that you might have missed otherwise. It's an essential step in ensuring the accuracy of your findings.

## Common EDA Techniques
1. **Univariate Analysis**: This involves the analysis of a single variable. For numerical variables, this might involve creating a histogram to view the distribution of the variable. For categorical variables, bar plots are used to count the frequencies of each category.

    - Numerical: The `plt.hist` function is used to generate a histogram. A histogram is a graphical representation that organizes a group of data points into a specified range. The data variable contains the data which the histogram will be built off of. The `bins` parameter tells you the number of bins that your data will be divided into. You can specify it as an integer or as a list of bin edges.
    - Categorical: The `sns.countplot` function is used to show the counts of observations in each categorical bin using bars. The `x` parameter is the categorical variable we want to count and `data` is the DataFrame where the variables reside.


```python
# For numerical variable
import matplotlib.pyplot as plt
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.hist(data, bins=4, alpha=0.5)
plt.show()

# For categorical variable
import seaborn as sns
tips = sns.load_dataset("tips")
sns.countplot(x='day', data=tips)
plt.show()
```


2. **Bivariate Analysis**: This involves the analysis of two variables to determine the empirical relationship between them. Scatter plots, correlation matrices, and cross-tabulations are commonly used techniques.

  - Scatter plot: The `sns.scatterplot` function is used to draw a scatter plot with possibility of several semantic groupings. The relationship between `x` and `y` can be shown for different subsets of the data using the `hue`, `size` and `style` parameters.
  - Correlation matrix: The `corr` function is used to compute pairwise correlation of columns, excluding NA/null values. The `sns.heatmap` function is used to plot rectangular data as a color-encoded matrix. This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the `ax` argument.

```python
# Scatter plot
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.show()

# Correlation matrix
corr = tips.corr()
sns.heatmap(corr, annot=True)
plt.show()
```

3. **Multivariate Analysis**: This involves the analysis of more than two variables to understand the relationships between them. Pair plots, 3D scatter plots, and parallel coordinate plots are some of the techniques used.

    - Pair plot: The `sns.pairplot` function is used to plot pairwise relationships in a dataset. By default, this function will create a grid of Axes such that each numeric variable in `data` will be shared across the y-axes across a single row and the x-axes across a single column.

```python
# Pair plot
sns.pairplot(tips)
plt.show()
```

4. **Summary Statistics**: These provide a quick summary of the data using measures such as mean, median, mode, standard deviation, etc.

      - The `describe` function generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding `NaN` values.

```python
# Summary statistics
tips.describe()
```

5. **Handling Missing Values**: Identifying and appropriately handling missing values is an important step of EDA.
      - The `isnull` function is used to detect missing values for an array-like object. The `sum` function then returns the sum of the missing values.

```python
# Checking for missing values
tips.isnull().sum()
```

6. **Outlier Detection**: Box plots and IQR (Interquartile Range) are commonly used to detect outliers.
    - The `sns.boxplot` function is used to draw a box plot to show distributions with respect to categories. A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables.
```python
# Box plot
sns.boxplot(x=tips['total_bill'])
plt.show()
```

Remember, the specific techniques used can vary depending on the nature of the dataset and the specific questions you are trying to answer. The above examples are written in Python using libraries like Matplotlib and Seaborn for visualization and Pandas for data manipulation.
## FAQs
  
### **Tell me about the largest data set you've worked with so far. What kind of data was it and how many variables and entries were involved?**
    - This question is more personal and depends on your past experience. You should mention the size of the dataset, the nature of the data (numerical, categorical, text, etc.), the number of variables, and the challenges you faced while working with such a large dataset.

### **What was your most challenging data analysis project?**
    - This question is also personal and depends on your past experience. You should mention a project where you faced significant challenges, how you overcame them, and what you learned from the experience.

### **Explain how you would estimate … ?**
    - This question is a bit vague without a specific context. However, it's usually about statistical estimation. For example, if you're asked to estimate the average height of students in a school, you might suggest taking a random sample of students, calculate the average height of the sample, and use that as an estimate of the average height of all students.

### **What is your process for cleaning data?**
    - This question is similar to the first one. You should describe the steps you take when you get a new dataset. This usually involves understanding the data, identifying and handling missing values, removing duplicates, converting data types, normalizing or scaling features, etc.

### **How do you explain technical concepts to a non-technical audience?**
    - This question tests your communication skills. You should mention how you use simple language, analogies, visuals, and real-world examples to explain complex technical concepts.

### **What are some spreadsheet concepts you think are important for a data analyst?**
    - Some important spreadsheet concepts for a data analyst might include formulas and functions, pivot tables, data filtering and sorting, conditional formatting, charting, etc.

### **What metrics would be critical to track for a company like [insert company name]?**
    - This question depends on the specific company and industry. However, some common metrics might include revenue, costs, profit margin, customer acquisition cost, customer lifetime value, net promoter score, etc.

### **Can you write a SQL query to [insert specific task]?**
    - This question depends on the specific task. However, here's an example of a SQL query that selects all records from a table where the value in the 'name' column is 'John':
    ```sql
    SELECT * FROM table WHERE name = 'John';
    ```
### **How would you use Python to analyze [insert specific data set or problem]?**
    - This question also depends on the specific data set or problem. However, you might mention how you would use pandas for data manipulation, matplotlib and seaborn for data visualization, scikit-learn for machine learning, etc. You might also write a short Python script that demonstrates how you would load a dataset and perform some basic analysis.
    ```python
    import pandas as pd
    df = pd.read_csv('data.csv')
    print(df.describe())
    ```
### **Working with Statistical Models**: Yes, statistical models are a key part of data analysis. For example, a linear regression model could be created in Python using the `statsmodels` library as follows:

```python
import statsmodels.api as sm
import pandas as pd

# Load a sample dataset
data = sm.datasets.get_rdataset('mtcars').data

# Define the dependent variable
y = data['mpg']

# Define the independent variables
X = data[['hp', 'wt']]

# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Fit the ordinary least-squares (OLS) model 
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
```

### **Favorite Step in Data Analysis**: 
Many data analysts enjoy the data modeling phase where they get to apply various statistical models and algorithms to the data to derive insights.

### **Knowledge of Statistics**: 
Statistics is fundamental to data analysis. It's used in everything from data collection to interpretation of results. For example, understanding statistical concepts like mean, median, mode, standard deviation, correlation, regression, hypothesis testing, etc., is crucial for a data analyst.

### **Motivation to Become a Data Analyst**:
Many people are drawn to data analysis because of the opportunity it provides to use analytical skills to solve real-world problems and make data-driven decisions.

### **Handling Missing or Corrupted Data**: 
This is typically handled through either deletion, imputation, or prediction. The choice depends on the nature of the data and the specific use case.

### **Data Visualization**: 
Data can be visualized using various tools like Matplotlib, Seaborn, or Tableau. For example, a bar plot can be created using Matplotlib in Python as follows:

```python
import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C']
values = [5, 7, 3]

plt.bar(categories, values)
plt.show()
```

### **Data Normalization**: 
This is the process of adjusting values measured on different scales to a common scale. It's often used in machine learning to ensure that all features have equal importance.

### **Process of Data Analysis**: 
The process typically involves several steps: defining the problem, collecting data, cleaning data, exploring data, modeling data, and communicating results.

### **Starting a New Project**: 
The process usually begins with understanding the project requirements, defining the problem statement, and planning the analysis approach.
