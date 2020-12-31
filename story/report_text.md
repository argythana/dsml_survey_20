# Data: where the truth lies ...
# Guitar or Drums?
# State of Machine Learning and Data Science 2020, Revisited

### Table of Contents
0. Overview
1. Key differences in Methodology
2. Key differences in Results
3. Data Scientist Demographic profile
4. Data Science & Machine Learning Experience
5. Employment Demographics
(6. Technology)
7. Conclusion
8. Appendix A: Spam and user error classification EDA methodology.
9. Appendix B: Suggestion for future Surveys.

## Overview

The Kaggle DS & ML Survey is an open online survey receiving thousands of responses from all over the world, offering unique insights.

Being a global online survey it is affected by a factor that Machine Learning is famous for mitigating: spam, and user error. According to the Survey Methodology, "spam" has been excluded from the data.

Using Exploratory Data Analysis (EDA) we find that without dropping possible outliers, a large part of the data should be excluded as spam or as having abnormal, irregular values.

We identify ... of such observations and after filtering them we get significantly different results than the "Executive Summary" and any other EDA, that does not filter the data.

Then, focusing on the issue of compensation, we explore the data accounting for the wide global diversity by grouping countries according to the World Bank "Income Groups" definition.

We also reconstruct the various classes (bins) in which the data are provided, accounting for their width and their underlying properties.

Since the weight of observations from the USA is disproportional, we exclude the USA from the global aggregates, offering a different view on the pronounced symmetric difference.

Combining these four features, our results diverge by many degrees from the "Executive Summary" ones, especially with regard to the issue of compensation levels.

Our findings add to the understanding of the data, providing information to students, professionals and interested companies in order to optimize their strategical behavior.

To assist with reproducing our methodology we have created a dedicated python "library" with functions, which may be parameterized and reused, in order to set various spam filtering thresholds and select different subsets of the data.

In Appendix A we explain out methodology in detail and in Appendix B, we provide another result of our analysis;
a comprehensive set of suggestions for future surveys.



## 1. Key differences in Methodology

### a) Filtering for general "pollution", spam and user error.

A key difference of our analysis is that the first part of our data exploration is dedicated to exploring abnormal/invalid observation values that have not be flagged as spam by the survey system, but should be excluded from the data nevertheless.

The number of observation unacceptable values and the criteria that had to be set, highlight the complexity and the paramount importance of exploring the data for invalid values and cleaning them before embarking on any other kind of analysis.

The criteria we use to clean the data are:

+ Participants who did not actually complete the survey but stopped after answering the first set of general demographic questions and no other relevant information that adds any value concerning DS & ML. We drop these irrelevant observations values as they offer no information whatsoever about any other in questions. A large part of these observations "completed" the Survey in less that 30 seconds.

```python  

code here
```

+ Participants that are too young for their experience (in Programming or Machine Learning). Obviously, it is impossible to be 24 years old or less and have 20+ years of experience.

```python  

code here
```

+ Participants whose salary, experience, age and country of residence are mutually exclusive. These criteria are adjusted for differences in country salary levels, using official data where available. As an example, it is impossible to be less than 21 with no experience whatsoever and earn above 500,000 usd, in any country in the world. Likewise, it is impossible to reside in the US, have more than 10 years experience in Programming and earn less than 1000 usd per year.

```python  

code here
```

As mentioned above, our classification rules are intentionally too lenient and could most probably be much stricter. This can be done in the spot, by setting a different parameter in the relevant dedicated filtering function. Nevertheless, even with minimum strictness we get significantly different results than any other EDA that uses the data as is.

For a detailed review of the various third party data sets and the methodology applied, please review the code submitted or read Appendix A.

The significance of these observations depends on the overall size of the subset they belong to and the metric that is calculated and they have a substantial impact on Survey results. For example, a salary of $500,000 outweighs 100 observations of salary of $5000. Similarly, dropping 10 observations from a range of 15 observations, means that the size of the category is one third of its initial unfiltered size.

In total, we drop ... observations and the new data set contains .... rows.

```python  

code here
```



### b) Adjusting for cross-country economic differences

The importance of meaningful information on compensation levels cannot be understated in a field that transcends national borders like few others. Nevertheless, the differences among different economies are so profound that a global distribution view on Data Scientist salary show an "everything goes" result. Besides a more accurate single country median calculation, we group the countries in the data according to the World Bank Income Groups and explore the distribution per group of countries.

This allows us in turn to explore the difference that experience levels have on salary in each countries income group.

```python

code here

```


### c) Reconstruction of data aggregated bins (classes)

Using different classes width for aggregating observation can be convenient and informative, but comes with a trade-off. A large amount of information is lost. This may lead to over or under representation of aggregated data depending on the relative width difference and the number of observations. This may lead to distortion of summarized information and optical "illusions" in visual representations

We reconstruct bins using various ways where we find this phenomenon to exist in the data and we gain new and different information.



### d) Explore the USA versus the Rest of the World symmetric differences

When comparing the aggregate metrics of a subset of the data with the overall aggregates we examine whether the subset in question influences heavily the aggregated values. For a subset with a low number of observations, it is informative for the subset to compare it to the total and find what how it measures against it.

For a subset with substantial weight this information is distorted by the fact that the subset defines the total to a significant degree. In this case, comparing the symmetric difference can add valuable information for both sides. Therefore, when comparing the USA, we exclude it from the total.



## Key differences in Results

The data point to significantly different results in salary distributions, salary median and experience levels among most countries. In contrast to the "Executive Summary", the data show that the median salary of participants from India is at least at the range of "...", double than the one in the official summary.

Grouping countries according to the World Bank Income Groups. To achieve this, we use official sources when available, we create a combined data set for average salaries and then by using the World Bank Country Income groups we examined the salary distribution in each group, by experience levels.

We reconstruct the various classes (bins) in which the data are provided, accounting for their width. Besides an important shift in the age distribution of participants, the data show that there is currently an acute global shortage of experienced professionals in Programming and Machine Learning, which will not change in the next five years, but maybe afterwards.

Fourthly, we chose a different approach in contrasting the differences of USA versus the world. Since the weight of USA observations is disproportional, when comparing the two we exclude USA from the global aggregates, offering a different view on the pronounced symmetric difference. This shows and even greater shortage of experienced Data Scientists in the Rest of the World (Row).

Combining these four features, and focusing on the other occupations as well, our insights diverge by many degrees to the Executive Summary results, especially with regard to the issue of DS & ML compensation levels. Those results add to the understanding of the data, providing information to students, professionals and interested companies in order to optimize their strategy


## Appendix A: Step by Step methodology of data set filtering.


### a) Identifying general "pollution", spam and user error.

Starting from ground zero, we examine the time that it took participant to complete the survey.

Checking duration to complete and number of questions answered, we identified a large set of observations that completed the survey in less than 30 seconds and only answered the first set of basic demographic questions and no other information that would add value concerning the DS & ML Survey. We decided to drop these observations as they offer no information whatsoever about the issues in question, are spam or irrelevant.

But, since meaningful differences in the results concerning DS & ML, originate in observations who spammed the answers instead of not answering them, we set logical thresholds for invalid values, on mutually exclusive value pairs and their combinations.

```
  df.time ...  
```

There are hundreds of observations below 30 seconds. It is impossible to complete a survey of this length in 20 seconds. We could set an arbitrary time threshold here, but we since this points to some people not completing the Survey, we decided there is another way to check this.

Conclusion 1: the spam system method includes participants which did not actually complete the Survey and answered whatever as fast as possible. We could use this criterion to drop many observations, but, we found an optimal one.

Studying the questionnaire we noticed that, as is usual with surveys, the first set of questions included general, demographic ones that could fit in any survey (such as age, gender, country). So, we decided to check which participants did not answer anything besides these question and the next one.

We decided to drop these observations, since they offer nothing to our understanding on DS & ML, other than general demographic variables of those who started the Survey. We could set a more strict spam criterion -e.g. at least 3 non-demographic answers-, but as a general methodological rule we chose have unambiguous lenient criteria for spam classification.

There are various types of irregular values in the data with value of age "18-21" and salary "> 500.000".
eg:

```  


```

Also, there are hundreds of observations in the US stating high experience with early salary below 1000$.

Many of these invalid values are in the same submission row, a fact that adds support to the decision to exclude them from the data. As a result, the number of observations that are dropped using each category of spam depends on the order of dropping it.

A characteristic example is:


Using official sources, we create an "average salary" threshold for each country. Then, to avoid excluding outliers, we set the "lower than average" threshold to be one level below the average level and two levels for lower income countries. using the lower than average thresholds, we exclude observations with salary lower than one third of the defined threshold, only for participants with experience more than 10 years. 
  
The significance of these observations depends on the overall size of the subset they belong to and the metric that is calculated.
For example, the average salary of someone with 500000$ outweighs 100 observations of salary of $ 5000.
Similarly, dropping 10 observations from a range of 15 observations, means that the size of this category is only one third of its initial unfiltered size.

In total, we drop ... observations and the new data set contains .... rows.









