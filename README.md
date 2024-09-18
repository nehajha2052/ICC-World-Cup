# ICC-World-Cup
### Dataset

The ICC Cricket World Cups - ODIs Master Data is an invaluable resource that meticulously documents the rich history of the Cricket World Cup from its inception in 1975 to 2023. This comprehensive dataset provides a detailed account of each match, capturing key details such as the tournament year, match venue, participating teams' batting order and scores, match outcomes, winning margins in terms of runs or wickets, and match classification, including league stage and quarter-finals. 

Furthermore, it sheds light on the hosting nation and the ultimate champion of each season, offering profound insights into the evolution and historical significance of cricket's premier event. With its wealth of information, this dataset serves as a treasure trove for cricket aficionados, analysts, and researchers seeking to delve into the cricketing history.

-----
### Data Source

The ICC Cricket World Cup dataset compiles the scores of every cricket match featured in the ODI World Cup spanning from 1975 to 2019. The ICC Cricket World Cup data used in this analysis was obtained from the Dataful website, specifically from the dataset available at the following link: https://dataful.in/datasets/5809/. Dataful is a platform that offers clean, structured, and readily accessible datasets covering various sectors related to India. These records were meticulously gathered from both the ICC website and Cricbuzz website, ensuring comprehensive coverage of the tournament's history.

Additionally, a supplementary CSV file named "WinRegions.csv" was generated to facilitate analysis, containing the names of participating countries along with their corresponding country codes.

#### Loading the Data

The analysis of the ICC World Cup dataset began with importing the cleaned dataset into RStudio. Initially stored in a Microsoft Excel XLSX file format, the dataset was transitioned to CSV format after export from a Google Doc/Sheet to facilitate smooth integration into the R environment. Subsequently, a thorough inspection of the data structure was conducted to verify its fidelity and integrity. This initial exploration provided a glimpse into the dataset's structure and allowed for a preliminary assessment of the variables and their respective values. It served as a crucial step in familiarizing ourselves with the data structure, dataset's contents and potential patterns for laying the groundwork for subsequent analyses and insights.

1. icc-cricket-world-cups.csv:

| Column Name  | Data Type  | Description |
| :------------ |:---------------:| -----:|
| ID      | Integer | Unique identifier for each record |
| Year      | Integer        |   The year in which the Cricket World Cup took place  |
| HostCountry      | Text | The country that hosted the Cricket World Cup |
| MatchType      | Text        |   The type of match, including Final, Semi-Final, or League  |
| MatchPeriod      | Text | The time period of the match such as Day or Day Night |
| FirstBattingCountry      | Text        |   The country that batted first in the match  |
| SecondBattingCountry      | Text | The country that batted second in the match |
| FirstBattingCountryScore      | Integer        |   The total score achieved by the first batting country  |
| SecondBattingCountryScore      | Integer | The total score achieved by the second batting country |
| WinningCountry      | Text        |   The country that won the match  |
| WCC      | Text | The abbreviation representing the winning country |
| Margin      | Integer        |   The margin of victory, either in runs or wickets  |
| MarginType      | Text | Indicates the type of margin, such as runs or wickets |
| Series      | Text        |   The series of matches, specifically the ICC Cricket World Cup  |
| SeasonWinner      | Text | The ultimate winner of the Cricket World Cup season |

2. WinRegions.csv

Additionally, a supplementary CSV file named "WinRegions.csv" was generated to facilitate analysis, containing the names of participating countries along with their corresponding country codes.

| Column Name  | Data Type  | Description |
| :------------ |:---------------:| -----:|
| WCC      | Text | Abbreviation for the contry | 
| country      | Text        |   Name of the participating country | 

-----
### Analysis
#### Temporal Evolution of Match Characteristics in the ICC Cricket World Cup
   
A detailed analysis of the temporal evolution of match characteristics within the context of the ICC Cricket World Cup is performed. By meticulously grouping the data by year and margin type, the code systematically calculates the count of unique match types and match periods for each year. Furthermore, it categorizes the tournament years into distinct periods based on predefined intervals, facilitating a more structured analysis. Leveraging the versatile capabilities of the ggplot2 package, the code meticulously
crafts visualizations to unveil insights into the dynamic evolution of match types, match periods, and margin types over time. 

These visualizations offer a compelling narrative, shedding light on the changing landscape of cricket matches within the Cricket World Cup framework. From the emergence of new match types to the shifts in match periods and the impact of different margin types on match outcomes, the code endeavors to provide a comprehensive understanding of the tournament's evolution. Through its visual exploration, the code aims to unravel intricate patterns and trends, enabling stakeholders to gain deeper insights into the historical progression and dynamics of the ICC Cricket World Cup.

![image](https://github.com/user-attachments/assets/0063ecad-897a-45b2-9af4-9823cc1f4cbd)

-----
#### Wins Across Different Series Types

The distribution of wins across different Series types in ICC world cup is explored. Initially, the code filters out incomplete data entries and groups the remaining records by both the winning country code (WCC) and the series type (Series). This grouping facilitates the counting of wins for each country within each series type, providing valuable insights into their performance. Subsequently, the summarized data is visualized using ggplot2, a powerful plotting library in R. The code generates a bar plot where each series type is represented on the x-axis, and the corresponding total number of wins is depicted on the y-axis. This graphical representation offers a succinct overview of how wins are distributed across various series types in ICC Cricket tournaments.

![image](https://github.com/user-attachments/assets/97af8878-936e-4b97-b600-0a6dabcb75b2)

-----
#### Total Number Of Wins By Each Participating Country

The visualization reveals significant insights into the performance of participating countries in ICC Cricket tournaments. Notably, Australia emerges as the standout performer with the highest number of wins overall. Following closely are India, New Zealand, and England, demonstrating comparable success in securing victories. Conversely, countries such as UAE, Canada, Netherlands, and Afghanistan exhibit the lowest win counts across the entire ICC Cricket World Cup timeline, indicating their relatively lower performance levels in the competition.

![image](https://github.com/user-attachments/assets/8e28da43-4e39-4053-a6c7-80ae844bde92)

-----
#### Number of wins by each country per year

To analyze and visualize the performance of different countries in ICC Cricket World Cups over time, the function calculate_total_wins_yearwise is used to calculate the total number of wins for each country by year. It iterates over each unique country, filters the data for that country, counts the number of wins for each year, and merges the results into a single data frame. 

Australia exhibited the most notable trajectory of change in the plotted data between 2000 and 2010, characterized by a rapid increase followed by a subsequent decline in performance during the early 2010s, albeit with some intermittent progress thereafter. Similarly, India displayed a comparable pattern, experiencing substantial growth in the mid-2000s but encountering a significant downturn in wins during the late 2000s. However, from 2010 onwards, India demonstrated the most noteworthy enhancement in performance among all participating nations. Conversely, South Africa has maintained a relatively stable progression since its emergence in the early 1990s. Sri Lanka demonstrated significant advancements in the mid-1990s and mid-2000s, yet experienced downturns both in the late 1990s and again in the late 2000s.

![image](https://github.com/user-attachments/assets/9f2f6d2d-eec6-4d37-8acd-b28c771f2839)

-----
#### Win density plotted in Map

The goal of this code block within the ICC Cricket Match project in R is to create a thematic world map displaying participating countries in ICC World Cup matches and color-code them based on their number of wins. The code begins by installing and loading necessary packages such as "sp" for spatial data processing and "rworldmap" for mapping functionality. It then filters the map data to include only countries that have participated in the ICC World Cup matches. The countries are color-coded according to the number of wins they have achieved, with different color scales representing different win thresholds. The code ensures that all participating countries are correctly matched with their corresponding win counts and assigns a gray color to any unmatched countries. Subsequently, the code plots all country borders and overlays the filtered map with the color-coded participating countries. Finally, it defines legend labels based on the assigned colors and plots the legend accordingly to provide a clear visual representation of the win distribution among participating countries. Through this process, the code facilitates the visualization of ICC World Cup wins on a global scale, enabling insights into the distribution and performance of countries in the tournament.

![image](https://github.com/user-attachments/assets/4b69854b-e146-42f6-8cf6-9700781c9f19)

-----
#### Wins Per Match Type

To analyze and visualize the distribution of wins among different countries across various match types in ICC Cricket World Cup tournaments we first extract unique countries and match types from the dataset and then iterate through each match type, the code calculates the number of wins for each country in that specific match type. This iterative process allows for a granular examination of how countries perform in different types of cricket matches, such as league matches, semi-finals, and finals. The resulting data frame is then reshaped to a long format to facilitate plotting. The line graph generated using ggplot presents a clear and comprehensive overview of the win distribution, with each country represented by a distinct line color. 

The analysis reveals a notable trend wherein the majority of countries have achieved a significant number of victories in league matches, followed closely by group matches. Specifically, Australia emerges as the most successful nation in league matches and finals, surpassing other countries in terms of overall wins. In league matches and finals, India ranks second in terms of victories, showcasing a competitive performance in these crucial match types. This pattern underscores the prominence of consistent performance and strategic prowess, particularly in long-standing league competitions and high-stakes final matches, within the landscape of international cricket tournaments.

![image](https://github.com/user-attachments/assets/83cfc021-19fc-4947-a026-2426d66cda64)

-----
#### Clustering Analysis of ICC Cricket World Cup Data

By employing the K-means clustering algorithm we analyze the dataset containing information about ICC Cricket World Cup matches to partition the dataset into a predetermined number of clusters, with our specific case involving the creation of three distinct clusters. Each match instance is then assigned a cluster label based on its feature values, allowing us to group similar matches together to visualize the grouping patterns and separability of data points which enables us to understand the inherent structure of the dataset. 

The silhouette score indicates the degree of separation between clusters, with higher values suggesting better-defined clusters. In this case, a silhouette score of 1.2809 indicates reasonably well-separated clusters. The WCSS represents the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value signifies tighter clusters and better overall clustering performance. With a WCSS of 524.8857, the algorithm demonstrates effective grouping of match instances based on their feature similarities.

![image](https://github.com/user-attachments/assets/90c67b83-bce0-4112-a597-d0ffb1dccd7e)

-----
#### Predictive Modeling of ICC Cricket World Cup Winners

The obtained root mean squared errors (RMSE) for the regression models are as follows:

      Polynomial Regression: 1.321359
      Spline Model: 1.316253
      Generalized Additive Model (GAM): 0.6010556
      Decision Tree: 2.592034
   
These RMSE values represent the average deviation between the predicted and actual ICC Cricket World Cup season outcomes. Lower RMSE values indicate better predictive accuracy. Among the models evaluated, the GAM model achieved the lowest RMSE, suggesting that it provides the most accurate predictions for ICC Cricket World Cup outcomes compared to the other models tested. The GAM model stands out with a significantly lower RMSE of approximately 0.60, indicating superior predictive accuracy compared to the polynomial regression and spline models. A lower RMSE suggests that the GAM model's predictions are closer to the actual outcomes, with less average deviation, making it a more reliable predictor for ICC Cricket World Cup season results. Regression models are developed and evaluated to predict the outcome of ICC Cricket World Cup seasons based on various features. 

First, the dataset is preprocessed by converting categorical variables to factors and removing unnecessary variables. Then, the data is split into training and testing sets. Four different regression models are built and evaluated: Polynomial Regression, Splines, Generalized Additive Models (GAM), and Decision Trees. These models are trained on the training data and their performance is assessed using accuracy as the evaluation metric. Once the best-performing model is identified based on accuracy, predictions are made for a hypothetical scenario using the chosen model. Finally, the root mean squared error (RMSE) is calculated for each model to compare their performance.

The decision tree analysis reveals nuanced insights into the determinants of success in cricket world cup matches. It underscores the pivotal role of features such as the host country, tournament year, and specific series formats in shaping match outcomes. For matches hosted by "Australia/New Zealand", "England/Ireland/Netherlands/Scotland", and "India/Pakistan", the model further splits based on the Series. For matches hosted by "Kenya/South Africa/Zimbabwe" and "West Indies", no further splits based on the host country or series are made, indicating that perhaps these host countries have less variability or fewer matches in the dataset. Here, the split based on the condition Year >= 2015 suggests potential shifts in trends or dynamics in recent cricket world cup competitions compared to earlier periods, while splits like Year >= 1981 hint at historical variations in match dynamics. he analysis underscores the influence of tournament series on match results, indicating varying outcomes across different cricket world cup editions. For example Australia's flawless records in the ‘ICC Cricket world Cup’, ‘ICC World Cup’, and ‘Reliance World Cup’ series. Conversely, in other series, competition among teams appears more evenly distributed. 

These insights can inform strategic decision-making processes for teams and cricket management, enabling them to tailor strategies, team compositions, and training regimens to maximize performance and success in future cricket world cup tournaments.

![image](https://github.com/user-attachments/assets/64403dba-f8a3-4dba-995c-12f225720789)

-----
#### Exploratory Data Analysis and Outlier Detection

The process begins by loading the dataset and exploring its structure and summary statistics to gain an initial understanding. The detection of outliers is then carried out using a function based on the interquartile range (IQR) method, which allows for the identification of data points significantly deviating from the typical range of values. Upon detecting outliers in the specified numerical variables, the code provides insights by displaying the number of outliers detected, their indices, and the outlier data itself. 

Furthermore, to facilitate a clearer understanding, the code generates three visualizations: one depicting outliers for FirstBattingCountryScore over Year, another for SecondBattingCountryScore over Year, and the third displaying outliers for FirstBattingCountryScore versus SecondBattingCountryScore. The presence of red points in these plots, particularly for the years 1979, 2003, and 2023, indicates observations with exceptionally high or low values, signifying potential exceptional match performances or anomalies warranting further investigation. 

Through this systematic approach to outlier detection and visualization, the code contributes to a comprehensive understanding of the dataset's characteristics and highlights noteworthy patterns or outliers that may influence subsequent analyses or decisions.

![image](https://github.com/user-attachments/assets/bfcf9900-d8a5-495c-a893-312c2ba45417)
![image](https://github.com/user-attachments/assets/f559c2c5-9f1f-467c-845c-690065bcee26)
![image](https://github.com/user-attachments/assets/b12d22ef-b8e6-431d-81c2-54b9517f8ebd)

-----
#### Statistical Analysis of Categorical Variables

This particular analysis is done to understand the association between some of the categorical data of ICC Cricket World Cups by using contingency table analysis and a chi square test of independence.

First, the code reads the CSV file containing match data and ensures that the variables used for mapping are treated as factors to facilitate categorical analysis. Next, it computes the contingency table, which tabulates the frequencies of matches won by each country across different years. This table is then converted into a data frame for plotting purposes. The code then generates a heatmap using ggplot2, where the x-axis represents the years, the y-axis represents the winning countries, and the color intensity indicates the frequency of wins. This visualization provides a quick overview of the distribution of wins over time and across countries.

Additionally, the chi-square test of independence is applied to assess whether there is a significant association between the year and the winning country variables. The test yields a test statistic (X-squared), degrees of freedom (df), and a p-value.

##### Match Type Vs. Season Winner

![image](https://github.com/user-attachments/assets/2516c4a9-633e-4ff3-adb1-98af334337c0)

    Pearson's Chi-squared test
    data: contingency_table
    X-squared = 264.82
    df = 60
    p-value < 2.2e-16

Since the p-value is much smaller than the significance level of 0.05, there is overwhelming evidence to reject the null hypothesis. This implies that there is a significant association between the "MatchType" and "SeasonWinner" variables. 

In practical terms, it indicates that the type of match (e.g., group stage, semi-final, final) is strongly related to which country emerges as the winner of the ICC Cricket World Cup in a particular season. The significant association suggests that different types of matches may have varying impacts on the probability of a specific country winning the tournament in a given season.

##### Match Type Vs. Winning Country

![image](https://github.com/user-attachments/assets/20b9a915-ba30-4e24-892a-24f1dfb79671)

    Pearson's Chi-squared test
      data: contingency_table
      X-squared = 146, 
      df = 180, 
      p-value = 0.9702

Since the p-value (0.9702) is greater than the significance level (0.05), there is insufficient evidence to reject the null hypothesis. This implies that there is no significant association between the "Match Type" and "WinningCountry" variables. 
In practical terms, it suggests that the type of match (e.g., group stage, semi-final, final) does not significantly influence the probability of a specific country winning the ICC Cricket World Cup.

This result indicates that regardless of the type of match, the likelihood of a country winning the tournament remains relatively consistent across different match types.

-----
#### Exploratory Data Analysis and Hypothesis Testing

To understand the relationship between the margin of victory (continuous variable) and the match period (categorical variable) statistical analyses are performed . The code aims to achieve the following objectives:

1. Explore the dataset by loading the ICC Cricket World Cups data and examining its structure and summary statistics
2. Visualize the distribution of the margin of victory across different winning countries using boxplots
3. Conduct hypothesis testing to investigate if there are significant differences in the margin of victory across different match periods
4. Calculate and interpret the results of the t-test, ANOVA, and F-test to determine the statistical significance of the observed differences.

Three types of statistical tests are performed:
a. T-test: It assesses whether the mean margin of victory differs significantly between different match periods
b. ANOVA (Analysis of Variance): It determines whether there are statistically significant differences in the mean margin of victory across multiple match periods
c. F-test (Variance Test): It examines whether there are significant differences in the variability (variance) of the margin of victory between different match periods.
![image](https://github.com/user-attachments/assets/3fbbaa58-bb9c-48f3-a5bf-2adea592e8b6)

a. print(t_test_result)

Welch Two Sample t-test

    data: Margin by MatchPeriod

      t = 1.2473
      df = 233.32
      p-value = 0.2135
      alternative hypothesis: 
          true difference in means between 
          group 'Day Night' and group 'Day' is not equal to 0
      95 percent confidence interval: -4.759068 21.182326

    sample estimates:
        mean in group 'Day Night': 53.69388
        mean in group 'Day': 45.48225

With a p-value greater than the significance level (commonly set at 0.05), we fail to reject the null hypothesis. This suggests that there is no significant difference in the mean margin of victory between Day Night and Day matches. The 95% confidence interval for the difference in means ranges from -4.759 to 21.182, indicating that the true difference is likely to fall within this range.

b. print(summary(anova_result))

|   | Df  | Sum Sq | Mean Sq | F value | Pr(>F) |
| :------------ |:---------------:| -----:| :------------ |:---------------:| -----:|
| Period      | 1 | 6908 | 6908 | 1.831 | 0.177 |
| Residuals      | 483        |   1822334 | 3773      |         |    |

Since the p-value is greater than 0.05, we do not reject the null hypothesis. This suggests that there are no significant differences in the mean margin of victory among the match periods ("Day Night" and "Day").
In other words, the match period does not have a significant impact on the margin of victory.

c. print(ftest_result)

    F test to compare two variances
    data: Margin by MatchPeriod

    F = 1.5184
    num df = 146
    denom df = 337
    p-value = 0.00214
    alternative hypothesis: true ratio of variances is not equal to 1
    95 percent confidence interval: 1.161559 2.015232

    sample estimates:
    ratio of variances: 1.518395

The p-value is less than 0.05, indicating statistical significance. Thus, we reject the null hypothesis, suggesting that there is a significant difference in the variance of the margin of victory between Day Night and Day matches. The ratio of variances is estimated to be 1.5184, indicating that the variance in margin of victory is higher in one of the match periods compared to the other.
Overall, these results suggest that while there may not be significant differences in the mean margin of victory across different match periods, there is evidence of variation in the variance of the margin of victory between Day Night and Day matches.

-----






