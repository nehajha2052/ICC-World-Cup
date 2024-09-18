library("gganimate")
library("data.table")
library("knitr")
library("gridExtra")
library("tidyverse")
library("plotly")

# LOAD ATHLETES EVENTS DATA

dataICC <- read_csv("data/icc-cricket-world-cups.csv", col_types = cols(
  ID = col_character(),
  Year = col_integer(),
  HostCountry = col_character(),
  MatchType = col_character(),
  MatchPeriod = col_character(),
  FirstBattingCountry = col_character(),
  SecondBattingCountry = col_character(),
  FirstBattingCountryScore = col_integer(),
  SecondBattingCountryScore = col_integer(),
  WinningCountry = col_character(),
  Margin = col_integer(),
  MarginType = col_character(),
  Series = col_character(),
  SeasonWinner = col_character()
)
)

glimpse(dataICC)
head(dataICC)
summary(dataICC)
------------------------------------------------------------------
# LOAD DATA MATCHING WCCs (WINNING COUNTRY CODE) WITH COUNTRIES

WCCs <- read_csv("data/WinRegions.csv", col_types = cols(
  WCC = col_character(),
  country = col_character()
))

glimpse(WCCs)
head(WCCs)
summary(WCCs)
str(WCCs)
------------------------------------------------------------------

# NUMBER OF MATCHTYPES WITH RESPECT TO MARGINTYPES
library(gridExtra)
num <- dataICC %>%
  group_by(Year, MarginType) %>%
  summarize(MatchTypes = length(unique(MatchType)), SeasonWinners = length(unique(SeasonWinner))
  )

num <- num %>%
  mutate(gap = ifelse(Year<=1996, 1, Year),
         ifelse(Year>1996 & Year<=2007, 2, Year),
         ifelse(Year>2007, 3, Year))

plotMatchTypes <- ggplot(num, aes(x=Year, y=MatchTypes, group=interaction(MarginType,gap), color=MarginType)) +
  geom_point(size=2) +
  geom_line() +
  scale_color_manual(values=c("green","red")) +
  labs(x = "", y = "MatchTypes", 
       title="Match Types w.r.t Margin Types", 
       subtitle = "ICC World Cup from 1975 to 2023")

grid.arrange( plotMatchTypes, ncol=1)

--------------------------------------------------------------
  
  # THE TOTAL NUMBER OF WINS IN EACH SERIES TYPE
  library(ggplot2)
  
  seriesWinC <- dataICC %>% 
  filter(!is.na(WinningCountry)) %>% 
  group_by(WCC, Series) %>%
  summarize(isWinningCountry = 1)

serieswinCo <- seriesWinC %>% 
  group_by(Series) %>%
  summarize(countryCount = sum(isWinningCountry))

serieswinCo  # Print the result

# Plotting the total number of wins in each series type
ggplot(serieswinCo, aes(x = Series, y = countryCount)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Total Number of Wins in Each Series Type",
       x = "Series Type",
       y = "Total Number of Wins") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

  ------------------------------------------------------------------
# ORDERING COUNTRY BY TOTAL WIN COUNT

  levelsCountry <- winCo %>%
  group_by(WinningCountry) %>%
  summarize(Total=sum(Count)) %>%
  arrange(Total) %>%
  select(WinningCountry) %>%
  slice(30:1)
------------------------------------------------------

# THE TOTAL NUMBER OF WINS BY EACH PARTICIPATING COUNTRY
  
library(ggplot2)
unq <- unique(dataICC$WinningCountry)

winning_countries <- dataICC$WinningCountry
unique_countries <- unique(winning_countries)
wins_count <- rep(0, length(unique_countries))

for (i in 1:length(unique_countries)) {
  wins_count[i] <- sum(winning_countries == unique_countries[i])
}

# Create a data frame with the result
result <- data.frame(Country = unique_countries, Wins = wins_count)

# Plotting the total number of wins by each participating country
ggplot(result, aes(x = Country, y = Wins)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Total Number of Wins by Each Participating Country",
       x = "Country",
       y = "Total Number of Wins") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

--------------------------------------------------------------
  plot_wins_progression <- function(csv_file) {
    # Read the CSV file
    data <- read.csv(csv_file, header = TRUE)
    
    # Convert Year column to factor
    data$Year <- as.factor(data$Year)
    
    # Convert WinningCountry column to factor
    data$WinningCountry <- factor(data$WinningCountry)
    
    # Aggregate wins by Year and WinningCountry
    wins_agg <- aggregate(WinningCountry ~ Year + WinningCountry, data = data, FUN = length)
    
    # Plot
    ggplot(wins_agg, aes(x = Year, y = WinningCountry, group = WinningCountry, color = WinningCountry)) +
      geom_line() +
      geom_point() +
      labs(x = "Year", y = "Number of Wins", title = "Number of Wins Progression by Winning Country") +
      theme_minimal() +
      guides(color = guide_legend(title = "Winning Country"))
  }


plot_wins_progression("data/icc-cricket-world-cups.csv")
------------------------------------------------------
  library(reshape2)
  calculate_total_wins_yearwise <- function(csv_file) {
  # Read the CSV file
  data <- read.csv(csv_file, header = TRUE)
  
  # Create an empty data frame to store the results
  result <- data.frame(Year = unique(data$Year))
  
  # Extract unique country names
  country_names <- unique(data$WinningCountry)
  
  # Iterate over each country to calculate the number of wins for each year
  for (country in country_names) {
    # Filter data for the current country
    country_data <- subset(data, WinningCountry == country)
    
    # Count the number of wins for each year
    wins <- aggregate(. ~ Year, data = country_data, FUN = length)
    
    # Rename the column to the country name and merge with the result
    colname <- paste("Wins_", country, sep = "")
    names(wins) <- c("Year", colname)
    result <- merge(result, wins, by = "Year", all.x = TRUE)
  }
  
  
  # Fill NA values with 0
  result[is.na(result)] <- 0
  
  return(result)
}
total_wins_yearwise <- calculate_total_wins_yearwise("data/icc-cricket-world-cups.csv")

# Remove columns with names containing "NA"
cleaned_table <- total_wins_yearwise[, !grepl("^NA", names(total_wins_yearwise))]

# Print the cleaned table
print(cleaned_table)
--------------------------------------------------

  plot_country_wins <- function(wins_table) {
    # Convert the table to long format
    wins_long <- tidyr::pivot_longer(wins_table, cols = -Year, names_to = "Country", values_to = "Wins")
    
    # Plot the data
    p <- ggplot(wins_long, aes(x = Year, y = Wins, color = Country)) +
      geom_line() +
      labs(title = "Number of Wins by Country Over Time",
           x = "Year",
           y = "Number of Wins",
           color = "Country") +
      theme_minimal() +
      theme(legend.position = "bottom") +
      scale_color_manual(values = rainbow(length(unique(wins_long$Country))))
    
    # Print the plot
    print(p)
  }
colnames(cleaned_table)[-1] <- gsub("Wins_", "", colnames(cleaned_table)[-1])
# Call the function with your wins table
plot_country_wins(cleaned_table)
-----------------------------------------------------------------------------------------  
  data <- read.csv("data/icc-cricket-world-cups.csv", stringsAsFactors = FALSE)
  result <- data.frame(Country = unique(data$WinningCountry))

  for (match_type in unique(data$MatchType)) {
    # Count the number of wins for each country in the current match type
    wins <- sapply(result$Country, function(country) {
      sum(data$WinningCountry == country & data$MatchType == match_type)
    })
    # Add the wins to the result data frame
    result[[match_type]] <- wins
  }
  
  result_long <- reshape2::melt(result, id.vars = "Country", variable.name = "MatchType", value.name = "Wins")
  
  
  # Plot the line graph
  ggplot(result_long, aes(x = MatchType, y = Wins, group = Country, color = Country)) +
    geom_line() +
    geom_point() +
    labs(title = "Number of Wins by Country for Each Match Type",
         x = "Match Type",
         y = "Number of Wins") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
  
-------------------------------------------------------------------------------------------  
 ###9. You may use clustering techniques to cluster your instances based on one or more features and develop discussions about individual clusters.
  
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggfortify) # for visualizing clustering results
  library(factoextra) # for additional clustering visualization and evaluation
  library(fpc) 
  data <- read_csv("data/icc-cricket-world-cups.csv")
  
  # Select relevant features for clustering
  features <- c('FirstBattingCountryScore', 'SecondBattingCountryScore', 'Margin')
  
  scaled_data <- scale(data[, features])
  
  # Apply K-means clustering
  k <- 3  # Number of clusters
  kmeans_result <- kmeans(scaled_data, centers = k)
  
  # Add cluster labels to the original data
  data$Cluster <- kmeans_result$cluster
  
  # Visualize the clusters
  autoplot(kmeans_result, data = scaled_data, frame = TRUE, frame.type = "norm") +
    labs(title = "K-means Clustering")
  
  # Analyze and discuss the clusters
  cluster_means <- data %>%
    group_by(Cluster) %>%
    summarize_at(vars(features), mean)
  
  print(cluster_means)
  
  # Additional visualization and evaluation (optional)
  fviz_cluster(kmeans_result, data = scaled_data, geom = "point", stand = FALSE, ellipse = TRUE, ellipse.type = "norm") +
    labs(title = "K-means Clustering Visualization")
  
  dissimilarity_matrix <- dist(scaled_data)
  
  # Evaluate cluster quality
  cluster_quality <- cluster.stats(dissimilarity_matrix, kmeans_result$cluster)
  print(cluster_quality)
  
  silhouette_score <- cluster::silhouette(kmeans_result$cluster, dist(scaled_data))
  print(paste("Silhouette Score:", mean(silhouette_score)))
  
  wcss <- kmeans_result$tot.withinss
  print(paste("WCSS:", wcss))
  
  TSS <- sum(apply(scaled_data1, 2, var)) * nrow(scaled_data1) 
  WCSS <- sum(kmeans_result$withinss)
  BCSS <- TSS - WCSS
  print(paste("BCSS:", BCSS))
  
  scaled_data
  # Visualize the clusters with cluster assignments
  autoplot(kmeans_result, data = scaled_data, frame = TRUE, frame.type = "norm", 
           label = TRUE, label.size = 3, label.color = "black", main = "K-means Clustering") +
    scale_color_manual(values = c("red", "green", "blue")) +  # Assign colors to clusters
    labs(title = "K-means Clustering") +
    theme_minimal()
  
 ------------------------------------------------------------------------------------------ 
  ###8.
    # Load libraries
  library(readr)
  library(caret)
  
  # Load the dataset
  data <- read_csv("data/icc-cricket-world-cups.csv")
  
  # Data preprocessing
  # Remove the 'ID' column
  data <- data[, !names(data) %in% c("ID")]
  
  # Convert categorical variables to factors
  data$MatchType <- as.factor(data$MatchType)
  data$MatchPeriod <- as.factor(data$MatchPeriod)
  data$FirstBattingCountry <- as.factor(data$FirstBattingCountry)
  data$SecondBattingCountry <- as.factor(data$SecondBattingCountry)
  data$WinningCountry <- as.factor(data$WinningCountry)
  data$MarginType <- as.factor(data$MarginType)
  data$Series <- as.factor(data$Series)
  data$SeasonWinner <- as.factor(data$SeasonWinner)
  
  str(data)
  # Split the data into train and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data$SeasonWinner, p = .8, 
                                    list = FALSE, 
                                    times = 1)
  train_data <- data[trainIndex, ]
  test_data  <- data[-trainIndex, ]
  
  # Ensure consistency of factor levels
  for (col in c("MatchType", "MatchPeriod", "FirstBattingCountry", 
                "SecondBattingCountry", "WinningCountry", 
                "MarginType", "Series", "SeasonWinner")) {
    levels(test_data[[col]]) <- levels(train_data[[col]])
  }
  
  # Choose a classification algorithm (Random Forest)
  model <- train(SeasonWinner ~ ., 
                 data = train_data, 
                 method = "rf")
  
  # Evaluate the model
  predictions <- predict(model, test_data)
  conf_matrix <- confusionMatrix(predictions, test_data$SeasonWinner)
  print(conf_matrix)
  
  # Perform hyperparameter tuning
  param_grid <- expand.grid(mtry = c(2, 3, 4))  # Example grid, adjust as needed
  tune_model <- train(SeasonWinner ~ ., 
                      data = train_data, 
                      method = "rf",
                      trControl = trainControl(method = "cv", number = 5),
                      tuneGrid = param_grid)
  
  # Predict using the tuned model
  predictions_tuned <- predict(tune_model, test_data)
  
  # Evaluate the predictions
  conf_matrix_tuned <- confusionMatrix(predictions_tuned, test_data$SeasonWinner)
  print(conf_matrix_tuned)
  
  
  # Get variable importance
  importance <- varImp(model)
  
  # Plot variable importance
  plot(importance)
  
  ------------------------------------------------------------------------
  ####7
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(caret)
  library(splines)
  library(mgcv)
  library(rpart)
  library(rpart.plot)
  
  # Load the dataset
  data <- read_csv("data/icc-cricket-world-cups.csv")
  
  # Preprocess the data
  # Convert categorical variables to factors
  cat_vars <- c("HostCountry", "MatchType", "MatchPeriod", "FirstBattingCountry", 
                "SecondBattingCountry", "WinningCountry", "MarginType", "Series", "SeasonWinner")
  data[cat_vars] <- lapply(data[cat_vars], as.factor)
  
  # Remove unnecessary variables like ID
  data <- select(data, -ID)
  
  # Split the data into train and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data$SeasonWinner, p = .8, 
                                    list = FALSE, 
                                    times = 1)
  train_data <- data[trainIndex, ]
  test_data  <- data[-trainIndex, ]
  
  # Convert SeasonWinner to numeric
  train_data$SeasonWinner <- as.numeric(train_data$SeasonWinner)
  
  # Polynomial Regression
  poly_model <- lm(SeasonWinner ~ poly(Year, 2) + poly(FirstBattingCountryScore, 2) + poly(SecondBattingCountryScore, 2), data = train_data)
  
  # Splines
  spline_model <- lm(SeasonWinner ~ ns(Year, df = 3) + ns(FirstBattingCountryScore, df = 3) + ns(SecondBattingCountryScore, df = 3), data = train_data)
  
  # Generalized Additive Models (GAM)
  gam_model <- gam(SeasonWinner ~ s(Year) + s(FirstBattingCountryScore) + s(SecondBattingCountryScore), data = train_data)
  
  # Decision Trees
  tree_model <- rpart(SeasonWinner ~ ., data = train_data, method = "class")
  
  # Plot decision tree
  rpart.plot(tree_model)
  
  # Evaluate model performance
  # For simplicity, let's use accuracy as the evaluation metric
  poly_pred <- predict(poly_model, test_data)
  poly_acc <- sum(poly_pred == test_data$SeasonWinner) / length(poly_pred)
  
  spline_pred <- predict(spline_model, test_data)
  spline_acc <- sum(spline_pred == test_data$SeasonWinner) / length(spline_pred)
  
  gam_pred <- predict(gam_model, test_data)
  gam_acc <- sum(gam_pred == test_data$SeasonWinner) / length(gam_pred)
  
  tree_pred <- predict(tree_model, test_data)
  tree_acc <- sum(tree_pred == test_data$SeasonWinner) / length(tree_pred)
  
  # Choose the best-performing model
  best_model <- ifelse(max(poly_acc, spline_acc, gam_acc, tree_acc) == poly_acc, "poly",
                       ifelse(max(poly_acc, spline_acc, gam_acc, tree_acc) == spline_acc, "spline",
                              ifelse(max(poly_acc, spline_acc, gam_acc, tree_acc) == gam_acc, "gam", "tree")))
  
  # Make predictions with the best model
  new_data <- data.frame(Year = 2025, FirstBattingCountryScore = 250, SecondBattingCountryScore = 240)
  
  best_pred <- switch(as.character(best_model),
                      "poly"   = predict(poly_model, newdata = new_data),
                      "spline" = predict(spline_model, newdata = new_data),
                      "gam"    = predict(gam_model, newdata = new_data),
                      "tree"   = predict(tree_model, newdata = new_data))
  
  # Print the prediction
  print(best_pred)
  
  # 1. Choose Evaluation Metrics
  # For regression, let's calculate RMSE
  rmse <- function(actual, predicted) {
    sqrt(mean((as.numeric(actual) - as.numeric(predicted))^2))
  }
  
  # 2. Calculate Evaluation Metrics
  poly_rmse <- rmse(test_data$SeasonWinner, poly_pred)
  spline_rmse <- rmse(test_data$SeasonWinner, spline_pred)
  gam_rmse <- rmse(test_data$SeasonWinner, gam_pred)
  tree_rmse <- rmse(test_data$SeasonWinner, tree_pred)
  
  # 3. Compare Models
  cat("RMSE for Polynomial Regression:", poly_rmse, "\n")
  cat("RMSE for Spline Model:", spline_rmse, "\n")
  cat("RMSE for GAM:", gam_rmse, "\n")
  cat("RMSE for Decision Tree:", tree_rmse, "\n")
  
  ------------------------------------------------------------------------------
###4
    
    # Load libraries
  library(readr)
  library(dplyr)
  library(ggplot2)
  
  # Load the data
  data <- read_csv("data/icc-cricket-world-cups.csv")
  
  # Explore the data
  str(data)
  summary(data)
  # Check distributions and relationships using plots
  # Example:
  # ggplot(data, aes(x = Year, y = FirstBattingCountryScore)) + geom_point()
  
  # Detect outliers (example using boxplots)
  detect_outliers <- function(x) {
    q1 <- quantile(x, 0.25)
    q3 <- quantile(x, 0.75)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    outliers <- which(x < lower_bound | x > upper_bound)
    return(outliers)
  }
  
  # Select numerical variables for outlier detection
  numerical_vars <- select(data, Year, FirstBattingCountryScore, SecondBattingCountryScore)
  
  # Detect outliers for each numerical variable
  outliers <- lapply(numerical_vars, detect_outliers)
  
  # Analyze outliers
  outlier_indices <- unlist(outliers)
  outlier_data <- data[outlier_indices, ]
  
  # Provide insights
  cat("Number of outliers detected:", length(outlier_indices), "\n")
  cat("Outlier indices:", outlier_indices, "\n")
  cat("Outlier data:\n")
  print(outlier_data)
  
  # Plot Original Data with Outliers for First Batting Country Score
  ggplot(data, aes(x = Year, y = FirstBattingCountryScore)) +
    geom_point(color = "blue", alpha = 0.5) +  # Blue points for original data
    geom_point(data = outlier_data, aes(x = Year, y = FirstBattingCountryScore), color = "red") +  # Red points for outliers
    geom_text(data = outlier_data, aes(label = Year), vjust = -0.5, hjust = 0.5, color = "red") +  # Add labels for outliers
    labs(title = "Original Data with Outliers",
         x = "Year",
         y = "First Batting Country Score") +
    theme_minimal()
  
  # Plot Original Data with Outliers for Second Batting Country Score
  ggplot(data, aes(x = Year, y = SecondBattingCountryScore)) +
    geom_point(color = "blue", alpha = 0.5) +  # Blue points for original data
    geom_point(data = outlier_data, aes(x = Year, y = SecondBattingCountryScore), color = "red") +  # Red points for outliers
    geom_text(data = outlier_data, aes(label = Year), vjust = -0.5, hjust = 0.5, color = "red") +  # Add labels for outliers
    labs(title = "Original Data with Outliers",
         x = "Year",
         y = "Second Batting Country Score") +
    theme_minimal()
  
  # Plot Original Data with Outliers for First vs. Second Batting Country Score
  ggplot(data, aes(x = FirstBattingCountryScore, y = SecondBattingCountryScore)) +
    geom_point(color = "blue", alpha = 0.5) +  # Blue points for original data
    geom_point(data = outlier_data, aes(x = FirstBattingCountryScore, y = SecondBattingCountryScore), color = "red") +  # Red points for outliers
    geom_text(data = outlier_data, aes(label = Year), vjust = -0.5, hjust = 0.5, color = "red") +  # Add labels for outliers
    labs(title = "Original Data with Outliers",
         x = "First Batting Country Score",
         y = "Second Batting Country Score") +
    theme_minimal()
  
  ------------------------------------------------------------------------
    # Load necessary libraries
  library(readr)
  library(dplyr)
  library(ggplot2)
  
  # Load the data from the CSV file
  data <- read_csv("data/icc-cricket-world-cups.csv")
  
  # Explore the data
  str(data)
  summary(data)
  
  # Visualize summary statistics of individual variables
  # Example: Boxplot of FirstBattingCountryScore
  ggplot(data, aes(x = "", y = FirstBattingCountryScore)) +
    geom_boxplot() +
    labs(title = "Summary Statistics of First Batting Country Score",
         y = "First Batting Country Score") +
    theme_minimal()
  
  # Visualize conditional statistics based on MatchType
  # Example: Barplot of average FirstBattingCountryScore by MatchType
  ggplot(data, aes(x = MatchType, y = FirstBattingCountryScore)) +
    geom_bar(stat = "summary", fun = "mean", fill = "skyblue") +
    labs(title = "Average First Batting Country Score by Match Type",
         x = "Match Type",
         y = "Average First Batting Country Score") +
    theme_minimal()
  
  # Visualize conditional statistics based on HostCountry
  # Example: Barplot of average SecondBattingCountryScore by HostCountry
  ggplot(data, aes(x = HostCountry, y = SecondBattingCountryScore)) +
    geom_bar(stat = "summary", fun = "mean", fill = "lightgreen") +
    labs(title = "Average Second Batting Country Score by Host Country",
         x = "Host Country",
         y = "Average Second Batting Country Score") +
    theme_minimal()
--------------------------  

    # Load required libraries
  library(gplots)
  library(RColorBrewer)
  
  # Create a data frame with countries and their win counts
  country_wins <- data.frame(
    Country = c('Australia', 'India', 'England', 'South Africa', 'New Zealand', 'Bangladesh',
                'Pakistan', 'Afghanistan', 'Netherlands', 'Sri Lanka', 'West Indies', 
                'Ireland', 'Zimbabwe', 'Canada', 'Kenya', 'UAE'),
    Wins = c(78, 63, 53, 45, 59, 16, 49, 5, 4, 42, 43, 7, 11, 2, 7, 1)
  )
  
  # Create a bar plot
  barplot(country_wins$Wins, 
          names.arg = country_wins$Country,
          col = "skyblue",
          main = "ICC World Cup Wins by Country",
          xlab = "Country",
          ylab = "Wins"
  )
-------------------------------------------------------------------------
  # Install and load required packages
  install.packages("sp")
  install.packages("rworldmap")
  library(sp)
  library(rworldmap)
  
  # Filter map data to include only participating countries
  filtered_map <- subset(newmap, NAME %in% country_wins$Country)
  
  
  # Color participating countries based on wins
  country_colors <- ifelse(country_wins$Wins >= 50, "brown",
                           ifelse(country_wins$Wins >= 40, "red",
                                  ifelse(country_wins$Wins >= 10, "pink", "yellow")))
  
  
  # Match country names
  matched_names <- match(filtered_map$NAME, country_wins$Country)
  
  # Set color for missing countries
  country_colors[is.na(matched_names)] <- "gray" 
  
  # Plot all country borders
  plot(newmap, xlim = c(-180, 180), ylim = c(-90, 90), axes = TRUE, border = "gray")
  
  # Plot the filtered map
  plot(filtered_map, col = country_colors[matched_names], add = TRUE)
  
  # Define legend labels based on the colors assigned
  legend_labels <- c("50+ Wins", "40-49 Wins", "10-39 Wins", "1-9 Wins")
  
  # Plot the legend with corrected labels
  legend("bottomleft", legend = legend_labels, fill = colors, title = "ICC World Cup Wins",
         cex = 0.7, bty = "n", inset = 0.02,
         box.lty = 0, box.lwd = 0, xjust = 0.05, yjust = 0.05,
         text.col = "black", horiz = TRUE)

-------------------------------------------------------------
 ####3
  # Step 1: Read the CSV file
  data <- read.csv("data/icc-cricket-world-cups.csv")
  
  # Ensure that the variables used for mapping are factors
  data$Year <- as.factor(data$Year)
  data$WinningCountry <- as.factor(data$WinningCountry)
  
  # Compute the contingency table
  contingency_table <- table(data$Year, data$WinningCountry)
  
  # Convert the table to a data frame for plotting
  contingency_df <- as.data.frame.table(contingency_table)
  
  # Plot the heatmap
  library(ggplot2)
  ggplot(contingency_df, aes(x = Var1, y = Var2, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "blue") +
    theme_minimal() +
    labs(title = "Contingency Table Heatmap",
         x = "Year",
         y = "WinningCountry",
         fill = "Frequency")
  
  
  
  # Step 4: Apply Chi-square test of independence
  chisq.test(contingency_table)
  
  #Result: 
  #Pearson's Chi-squared test
  #data:  contingency_table
  #X-squared = 246.89, df = 180, p-value = 0.0006905
  
  #The result of the chi-square test indicates that there is a significant association between the "Year" and "WinningCountry" variables at a significance level of 0.05. Here's how to interpret the output:
  
  #Pearson's Chi-squared test is a statistical test used to determine whether there is a significant association between two categorical variables.
  #The test statistic (X-squared) is calculated as 246.89.
  #The degrees of freedom (df) is 180, which is calculated as (number of rows - 1) * (number of columns - 1).
  #The p-value associated with the test is 0.0006905.
  #Interpretation:

  #Since the p-value (0.0006905) is less than the significance level (typically 0.05), we reject the null hypothesis.
  #Therefore, we conclude that there is a significant association between the "Year" and "WinningCountry" variables.
  #In practical terms, this means that the year in which a cricket match occurred is significantly associated with the country that won the match. Further analysis or exploration may be warranted to understand the nature of this association, such as examining specific years or countries in more detail.
-----------------------------------------------------------------------------------
  ###2    
  data <- read.csv("data/icc-cricket-world-cups.csv")

  head(data)
  summary(data)
  
  library(ggplot2)
  
  # Assuming "Year" as the categorical variable and "Margin" as the continuous variable
  ggplot(data, aes(x = as.factor(WinningCountry), y = Margin)) +
    geom_boxplot() +
    labs(x = "WinningCountry", y = "Margin") +
    theme_minimal()
  
  # T-test
  t_test_result <- t.test(Margin ~ MatchPeriod, data = data)
  
  # ANOVA
  anova_result <- aov(Margin ~ MatchPeriod, data = data)
  
  # F-test
  ftest_result <- var.test(Margin ~ MatchPeriod, data = data)
  
  print(t_test_result)
  print(summary(anova_result))
  print(ftest_result)
  
  -----------------------------------------

  library(ggplot2)
  
  # Read the CSV file
  data <- read.csv("data/icc-cricket-world-cups.csv")
  
  # Remove quotes from character columns
  data <- data %>% 
    mutate(across(where(is.character), ~gsub("'", "", .)))
  
  # Convert Margin column to numeric and remove commas
  data$Margin <- as.numeric(gsub(",", "", data$Margin))
  
  # Plotting
  p <- ggplot(data, aes(x = WinningCountry, y = Margin, fill = WinningCountry)) +
    geom_boxplot() +
    labs(title = "Box Plot of Margin by Winning Country",
         x = "Winning Country",
         y = "Margin") +
    theme_minimal()
  
  # Save the plot to a file
  ggsave("boxplot_margin_by_country.png", p)
  
  # Check if the file was saved successfully
  file.exists("boxplot_margin_by_country.png")