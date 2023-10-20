# OnlineNewsPopularity

# **Table Of Contents** 
[1	Introduction	2](#_toc148265303)

[1.1	Problem Statement	2](#_toc148265304)

[2	Data Composition and Exploration	2](#_toc148265305)

[2.1	Features present in the dataset	2](#_toc148265306)

[2.2	Data Issues	2](#_toc148265307)

[2.3	EDA	3](#_toc148265308)

[2.3.1	Target: ‘ shares’	3](#_toc148265309)

[2.3.2	Correlation analysis	3](#_toc148265310)

[2.3.3	The Weekend Reading Dip in News Consumption	4](#_toc148265311)

[2.3.4	World News Dominates Readers’ Preferences	4](#_toc148265312)

[2.3.5	Avg_positive_polarity vs avg_negative_polarity	5](#_toc148265313)

[2.3.6	Insights from scatter plots	6](#_toc148265314)

[3	Feature Selection	7](#_toc148265315)

[3.1	K Best Features	7](#_toc148265316)

[3.2	Recursive Feature elimination	7](#_toc148265317)

[4	Model Building and Evaluation	8](#_toc148265318)

[4.1	Modeling and Evaluation Strategy	8](#_toc148265319)

[4.1.1	Training Setup:	8](#_toc148265320)

[4.1.2	Model Evaluation Setup:	8](#_toc148265321)

[4.1.3	Analysis of Model Performances:	8](#_toc148265322)

[4.2	Results	9](#_toc148265323)

[5	Conclusion	9](#_toc148265324)

[6	References	10](#_toc148265325)











1. # <a name="_toc148265303"></a>Introduction
In the contemporary digital landscape, online news articles play a crucial role in disseminating information to a global audience. Understanding the factors influencing the popularity of these articles holds significant value for content creators and publishers. This project seeks to address this challenge by employing machine learning regression techniques to predict the popularity of online news articles based on a comprehensive set of attributes associated with each article.
1. ## <a name="_toc148265304"></a>Problem Statement

The objective of this project is to develop a machine learning regression model capable of predicting the popularity of news articles based on their inherent attributes. Specifically, our goal is to ascertain and quantify the determinants contributing to an article's popularity, including content quality, timing of publication, and social engagement. We aim to construct a predictive model that can estimate the expected level of popularity for new articles.

1. # <a name="_toc148265305"></a>Data Composition and Exploration

The dataset utilized for this project was obtained from the UC Irvine Machine Learning Repository and comprises 39,644 rows and 61 columns.
1. ## <a name="_toc148265306"></a>Features present in the dataset
As illustrated in Figure 1 below, all features, except for the initial 'url' column, are of type float or int. This implies that the categorical variables have been encoded before making the dataset available. 

*Figure 1 Datatypes present in dataset*
1. ## <a name="_toc148265307"></a>Data Issues
Various data issues were investigated, including checking for null values and duplicated rows, to determine if any data cleaning is required.

There are no null values in our dataset, obviating the need for null value handling. We also have no duplicated values present in our dataset.

Thus, it can be inferred that data quality is good and upholds data integrity.
1. ## <a name="_toc148265308"></a>**EDA**
The ‘url’ and ‘timedelta’ features are dropped as they are non-predictors and would not affect our model. 
1. ### <a name="_toc148265309"></a>**Target: ‘ shares’**
Summarizing our target variable, it was found that an average of 3395.380184 shares were made of the articles, further almost 75% of our data was below 2800 shares which might imply the presence of a significant number of outliers which may affect our decision.

As seen, in fig.2, the presence of outliers, in shares may affect our decision-making process and hence will be removed from our dataset.
1. ### <a name="_toc148265310"></a>**Correlation analysis** 
Figure 3: This figure illustrates the correlation between variables. From the figure, it becomes evident that, aside from the strong correlations among variables with themselves, there are concerns regarding multicollinearity. Notably, some variables exhibit strong correlations with each other. One significant example is the correlation between “n\_unique\_tokens” and "n\_non\_stop\_unique\_tokens" and "n\_non\_stop\_words," indicating redundancy.

Additionally, we observe that our target variable ‘shares’ has an average correlation of 0.27 with the rest of the features. As a result, it is imperative to explore alternative techniques for feature selection.

*Figure 3 Heatmap*
1. ### ` `**<a name="_toc148265311"></a>The Weekend Reading Dip in News Consumption**
From the graph, we notice a peak in the number of shares on Wednesday, followed closely by Tuesday and Thursday. Intriguingly, the weekends, typically a time when most individuals are free from work commitments, report the lowest share numbers of the week.









1. ### <a name="_toc148265312"></a>**World News Dominates Readers’ Preferences** 
Analysis of the data, in conjunction with the accompanying bar graph (see figure 5), reveals a compelling trend in news sharing preferences. Notably, news articles centered on world affairs emerged as the most frequently shared content, outpacing other data channels. Following closely were tech related news and entertainment news.













1. ### <a name="_toc148265313"></a>**Avg\_positive\_polarity vs avg\_negative\_polarity**

The two key polarity factors, avg\_positive\_polarity and avg\_negative\_polarity were represented on a scatter plot.

For avg\_positive\_polarity, it was evident that articles with polarity values ranging from 0.0 too 0.6 showed a wide range of shares. However, once the polarity exceeded 0.6, there was a notable decline in the number of shares. This suggests that articles with moderately positive language are more likely to be shared more than the rest.

Conversely, In the case of "avg\_negative\_polarity," it was observed that the number of shares increased as the polarity value moved closer to 0. In other words, articles with less negative language tended to attract a higher number of shares. This suggests that a milder use of negative language is associated with increased sharing of articles.














Also, when a scatter plot of global\_rate\_positive\_words and global\_rate\_negative\_words was plotted with respect to shares above 5000, it was observed that the highest shared articles lay between the range of 0.00 to 0.06 for positive words and 0.00 to 0.03 for rate of negative words.











1. ### <a name="_toc148265314"></a>**Insights from scatter plots** 
Scatter plots for each of the specified features were created. Noticed some interesting trends. By referring to the accompanying images for each feature (Figures 9-11) I made the following observations;

1. N\_tokens\_content (Figure 8); There was a range of shares when the content length varied between 0 and 2000 words.
1. N\_unique\_tokens (Figure 8); Shares showed variability within the range of 0.2 to 0.8 for tokens.
1. Num\_hrefs (Figure 9) and num\_links\_mashable (Figure 8); The number of shares followed a                                                        
1. pattern up to 50 links but then started declining.
1. ` `Num\_self\_hrefs (Figure 9); Most of the data concentrated in the range of 0 to 25 self hrefs suggesting a trend.
1. ` `Num\_imgs (Figure 9); Shares had a correlation with approximately, up to 40 images but beyond that point they declined.


*Figure 9 Sactter plot 2*

1. # <a name="_toc148265315"></a>**Feature Selection**

The data was divided into training and testing using a 70-30 split ratio and then standardized to prevent data leakage. 
As previously mentioned, poor correlation between the independent variables and our target variable, forced us to look for alternative solutions for feature selection. Out of which two methods were chosen to select features for the model. They were:
1. ## <a name="_toc148265316"></a>**K Best Features**
Using the k best features module from the sklearn library,  the ten best features for model training were:

'data\_channel\_is\_entertainment','data\_channel\_is\_socmed', 'data\_channel\_is\_tech' ,'data\_channel\_is\_world' , kw\_min\_avg', 'kw\_avg\_avg', 'weekday\_is\_saturday', 'is\_weekend', 'LDA\_02', ’LDA\_04’
1. ## <a name="_toc148265317"></a>**Recursive Feature elimination**

Similarly, performing RFE with SVR as the model to get the features, we got the following features as the best 10 features.

" n\_tokens\_content", " data\_channel\_is\_socmed", " data\_channel\_is\_tech", " kw\_avg\_max", " kw\_max\_avg", " kw\_avg\_avg", " is\_weekend",  " LDA\_01", " LDA\_02",          " LDA\_03"




1. # <a name="_toc148265318"></a>**Model Building and Evaluation**

1. ## <a name="_toc148265319"></a>**Modeling and Evaluation Strategy**

1. ### <a name="_toc148265320"></a>**Training Setup:**
Training and testing set size: The dataset was split into a 70-30 train-test ratio.

` `Sampling procedure: The data was standardized to prevent data leakage.
1. ### <a name="_toc148265321"></a>**Model Evaluation Setup:**
Key Performance Indicators (KPI): Mean Squared Error (MSE) and R-squared (R²) were chosen as KPIs. MSE measures the prediction error's square magnitude, while R² quantifies the variance explained by the model.

` `Justification: MSE helps gauge the predictive accuracy, while R² assesses the model's goodness of fit to the data.

` `Success Thresholds: A low MSE indicates minimal prediction error, and a higher R² suggests a better fit.

Justification: These KPIs align with the potential application's requirement for accurate and interpretable predictions.
1. ### <a name="_toc148265322"></a>**Analysis of Model Performances:**
Three models were considered for training the dataset:  Linear Regression, Lasso Regression, and Random Forest Regressor.
1. #### ***Linear Regression***
Linear regression model was applied on the features obtained by K best features module and it gave the following results which were extremely poor.

` `To prevent this and because of the lack of hyperparameters in Linear regression model, regularization techniques were used with the help of Lasso regression.
1. #### ***Lasso regression***
Applying Lasso regression, with the alpha value of 7 the following results prevailed:

There was slight improvement as compared to linear regression but overall the model was still performing poorly. After cross validating the results, it was inferred that no overfitting is occurring as consistent values were observed for the same. 
1. #### ***Random Forest Regressor***
The best performing model of all was the random forest regressor, which performed as follows with these hyperparameters: n\_estimators=100, bootstrap=True,max\_depth=10,random\_state=42.

An extremely poor result in terms of r2 score was found out but Random Forest regressor managed to reduce the MSE by a large extent as compared to the other models used.

After applying grid search CV to tune hyperparameters. The results came out to be the following:

The R² value of approximately 0.013 suggests that only a minimal fraction of the variance in the target variable can be explained by the model. In this case, a higher R² value would have indicated a better fit to the data. The low R² value implies that the model's independent variables have limited explanatory power, and the model's predictions do not capture much of the target variable's variation.

1. ## <a name="_toc148265323"></a>**Results** 
Among the three models, Lasso Regression performed the best with MSE(1,072,219.32) and highest R2 (0.11). It indicates that lasso regression was able to provide a relatively better fit to the data as compared to Linear regression and random forest regressor even though random forest had a much lower MSE as compared to Lasso. 

1. # <a name="_toc148265324"></a>**Conclusion**
Throughout the analysis of this project, it was found that 

- Multiple variables exhibited diverse relationships with shares.
- The content length, content sentiment, and data channel significantly influenced sharing behavior.
- Understanding these relationships could be leveraged for feature selection and engineering to improve the model's predictive capabilities.

In conclusion, Lasso Regression was the top-performing model in predicting shares. However, future work should focus on model optimization through hyperparameter tuning and dimensionality reduction, considering the insights from EDA to further enhance predictive accuracy and utility.


1. # <a name="_toc148265325"></a>**References**
- Fernandes,Kelwin, Vinagre,Pedro, Cortez,Paulo, and Sernadela,Pedro. (2015). Online News Popularity. UCI Machine Learning Repository. https://doi.org/10.24432/C5NS3V.
- Indeed Career Guide. (n.d.). How to Conduct Exploratory Data Analysis. https://www.indeed.com/career-advice/career-development/how-to-conduct-exploratory-data-analysis
- GeeksforGeeks. (n.d.). Exploratory Data Analysis (EDA) - Types and Tools. https://www.geeksforgeeks.org/exploratory-data-analysis-eda-types-and-tools/
- Medium. (n.d.). Good Documentation for Machine Learning: A Guide. https://medium.com/geekculture/good-documentation-for-machine-learning-a-guide-93ebbb4c4ea
- Profiling. (n.d.). Profiling Documentation. https://docs.profiling.ydata.ai/4.5/SS
- scikit-learn. (n.d.). scikit-learn: Machine Learning in Python. https://scikit-learn.org/stable/index.html
- Machine Learning Mastery. (n.d.). Train-Test Split for Evaluating Machine Learning Algorithms. https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
- Machine Learning Mastery. (n.d.). Feature Selection in Python with scikit-learn. https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
- Towards Data Science. (n.d.). ANOVA for Feature Selection in Machine Learning. https://towardsdatascience.com/anova-for-feature-selection-in-machine-learning-d9305e228476
- GeeksforGeeks. (n.d.). Detect and Remove the Outliers using Python. https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
- Matplotlib. (n.d.). Matplotlib - Pyplot Tutorial. https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py

10 | Page

