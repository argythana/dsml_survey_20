# Data: where the truth lies ...
# Guitar or Drums?
# State of Machine Learning and Data Science 2020 Revisited

### Table of Contents
0. Overview
1. Key differences in Methodology
2. Key differences in Results
3. Data Scientist Demographic profile
4. Data Science & Machine Learning Experience
5. Employment Demographics
6. Technology
7. Conclusion
8. Appendix A: spam and user error classification EDA methodology.
9. Appendix B: Suggestion for future Surveys.

## Overview

The Kaggle DS & ML Survey is an open online survey receiving thousands of responses from all over the world. In its uniqueness if offers valuable insights to all interested parties.

Being a global online survey it is affected by a factor that Machine Learning is famous for mitigating: spam and user errors. According to the "Kaggle Survey Methodology", respondents that were flagged by the survey system as "spam" have been excluded from the results.

Using Exploratory Data Analysis (EDA) we show that a significant part of the data can still be classified as spam, without dropping any observations that might be outliers.  We identify ... of such observations and by filtering them  we work on a data set of ... observations.

Filtering out these observations we get significantly different results from the official "Executive Summary". In contrast to the "Executive Summary", the date show that the median salary of Survey participants is at least at the range of "" which is double than the one reported in the official summary. The data point to significantly different results in salary distributions and experience levels among countries.

Furthermore, we explore the data by controlling for the wide global diversity that exists among country salary levels. To achieve this, we used all available official sources for salaries, we created a combined data set for average salaries and then by using the World Bank Country Income groups we examined the salary distribution in each group, by experience levels.

Thirdly, we reconstruct the various classes (bins) in which the data are provided, adding an extra perspective on understanding current and future trends. The Survey data show that currently there is an acute global shortage of experienced  professionals in Machine Learning and Coding which will not change in the next 5 years, but afterwards.

Fourthly, we chose a different approach in contrasting the USA versus Global differences. Since US residents weights on all relating to DS & ML is disproportional, when comparing the two we exclude USA from the global aggregates, offering a view on their pronounced symmetric difference.

Combining the four above features, our insights add to the understanding of the data, providing clues to students, professionals and interested companies in order to optimize their strategy.

To assist interested parties with reproducing our methodology we have created functions, which may be parameterized and reused for setting various spam filtering thresholds.

In the Appendix, we also provide a comprehensive set of suggestions for future Kaggle online surveys.


## Key differences in Methodology

The first key difference is that the first part of our EDA is dedicated to exploring irregular observation values that have not be flagged as spam by the survey system and assess whether they may have a substantial impact on the results.

Starting from ground zero, we examine the time that it took participant to complete the survey.

```
  df.time ...  
```

Hundreds of responses below 30 seconds. It is impossible to complete a survey of this length in 20 seconds.
We could set an arbitrary time threshold here, but we since this points to some people not completing the Survey, we decided there is another way to check this.

Conclusion 1: the spam system method includes participants which did not actually complete the Survey, in fact they only answered the first demographic questions and nothing that adds value concerning DS & ML.

We decided to drop these observations, since they offer nothing to our understanding on DS & ML, other than general demographic variables of those who started the Survey. We could set a more strict spam criterion -e.g. at least 3 non-demographic answers-, but as a general methodological rule we chose have unambiguous lenient criteria for spam classification.

More importantly, meaningful differences in the results concerning DS & ML, originate in observations who spammed the answers instead of not answering them.

Therefore, using only EDA methods we set logical thresholds for invalid values, on mutually exclusive value pairs and their combinations.

These are:

+ "Age" and "Experience" (Programming or Machine Learning).

Obviously, it is impossible to be 24 years old or less and have 20+ years of experience.

+ "Salary" and "Experience", "age". This criteria in order to be lenient have been adjusted for differences in country salary levels.

The significance of these observations depends on the overall size of the subset they belong to and the metric that is calculated.
For example, the average salary of someone with 500000$ outweighs 100 observations of salary of $ 5000.
Similarly, dropping 10 observations from a range of 15 observations, means that the size of this category is only one third of its initial unfiltered size.

In total, we drop ... observations and the new data set contains .... rows.

For a detailed review of the methodology applied, please read Appendix A.


## Appendix A


There are various types of irregular values in the data with value of age "18-21" and salary "> 500.000".
eg:
```  


```

Also, there are hundreds of observations in the US stating high experience with early salary below 1000$.

Many of these invalid values are in the same submission row, a fact that adds support to the decision to exclude them from the data. As a result, the number of observations that are dropped using each category of spam depends on the order of dropping it.

A characteristic example is:   
  
The significance of these observations depends on the overall size of the subset they belong to and the metric that is calculated.
For example, the average salary of someone with 500000$ outweighs 100 observations of salary of $ 5000.
Similarly, dropping 10 observations from a range of 15 observations, means that the size of this category is only one third of its initial unfiltered size.

In total, we drop ... observations and the new data set contains .... rows.









