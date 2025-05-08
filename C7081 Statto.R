## Statistics C7081###
## Sarah-Jane  Childs
## Statistical Report on California house prices
## 08-05-2025

## CONTENTS ####
## 00 Setup
## 01 EDA
## 02 Linear regression
## 03 Multiple linear regression
## 04 Random forest

## 00 Setup #### 
# Colab is so helpful that this takes 25 minutes :(

install.packages(c("tidymodels", "tidyverse", "ranger", "vip", "spdep",
                   "recipes", "tune", "rsample", "ggcorrplot", "glmnet",
                   "moments", "viridis", "ggmap", "e1071", "sf", "sp", "pdp"))

# If it's your first time with tidymodels, you may get an error message due to 
# your 'cli' folder being out of date. Manually remove it (R\win-library\ etc) 
# and then install.packages("cli", version = ">= 3.6.1")

# List and load
libraries <- c("tidymodels", "tidyverse", "ranger", "vip", "spdep",
               "recipes", "tune", "rsample", "ggcorrplot", "glmnet",
               "moments", "viridis", "ggmap", "e1071", "sf", "sp", "pdp")

lapply(libraries, library, character.only = TRUE)


# Load data
url <- "https://raw.githubusercontent.com/sarahjanechilds/C7081-Statistics/refs/heads/main/California_Houses.csv"
df <- read.csv(url, sep = ",")


## 01 EDA ####

## 01 EDA ####
glimpse(df) 

# Histogram for target variable Median House Value
ggplot(df, aes(x = Median_House_Value)) +
  geom_histogram(binwidth = 50000, fill = 'steelblue', color = 'darkgrey') +
  theme_minimal() +
  labs(title = 'Distribution of Median House Value')

# Histogramos for all
df %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = 'steelblue', bins = 30) +
  facet_wrap(~ variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "Histograms of Features")

# Add new variables
# Create population_density and rooms_per_house variables
df <- df %>%
  mutate(Population_Density = Population / Households,
         Rooms_per_House = Tot_Rooms / Households)

# Check summary details
summary(df)

## Deal with the outliers
# Select non-spatial variables
non_spatial_vars <- df %>% select(Median_House_Value, Median_Income, Median_Age,
                                  Tot_Rooms, Tot_Bedrooms, Population, Households, 
                                  Population_Density, Rooms_per_House, Distance_to_coast,
                                  Distance_to_LA, Distance_to_SanDiego, Distance_to_SanJose, 
                                  Distance_to_SanFrancisco)

# Initialize a list to store the number of outliers for each variable
outlier_counts <- list()

# Iterate through each non-spatial variable and apply the IQR method
for (var in colnames(non_spatial_vars)) {
  Q1 <- quantile(df[[var]], 0.25)
  Q3 <- quantile(df[[var]], 0.75)
  IQR <- Q3 - Q1
  
  outliers <- df %>%
    filter(df[[var]] < (Q1 - 1.5 * IQR) | df[[var]] > (Q3 + 1.5 * IQR))
  
  outlier_counts[[var]] <- nrow(outliers)
  # Cap the outliers
  df[[var]] <- ifelse(df[[var]] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR, df[[var]])
  df[[var]] <- ifelse(df[[var]] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, df[[var]])
}


## Deal with the skewness
# Calculate skewness for each non-spatial variable
skewness_values <- sapply(df %>% select(Median_House_Value, Median_Income, Median_Age,
                                        Tot_Rooms, Tot_Bedrooms, Population, Households, 
                                        Population_Density, Rooms_per_House, Distance_to_coast,
                                        Distance_to_LA, Distance_to_SanDiego, Distance_to_SanJose, 
                                        Distance_to_SanFrancisco), e1071::skewness)

# Print skewness values
print(skewness_values) 



# Apply square root transformation to specified columns
df <- df %>%
  mutate(
    Median_House_Value = sqrt(Median_House_Value),
    Tot_Rooms = sqrt(Tot_Rooms),
    Tot_Bedrooms = sqrt(Tot_Bedrooms),
    Population = sqrt(Population), 
    Distance_to_coast = sqrt(Distance_to_coast))

# Standardization of non-spatial features
standardized_recipe <- recipe(Median_House_Value ~ Median_Income + Median_Age +
                                Tot_Rooms + Tot_Bedrooms + Population + Households +
                                Population_Density + Rooms_per_House + Distance_to_coast +
                                Distance_to_LA + Distance_to_SanDiego + Distance_to_SanJose + 
                                Distance_to_SanFrancisco, data = df) %>%
  step_normalize(all_predictors())

# Apply the recipe to the data
standardized_data <- prep(standardized_recipe, training = df) %>%
  bake(new_data = NULL)

# Select the spatial features from the original dataframe
spatial_features <- df %>% select(Longitude, Latitude)  

# Combine standardized non-spatial features with spatial features
combined_data <- bind_cols(standardized_data, spatial_features)

df<-combined_data


# Reorder columns so that Median_House_Value appears first
df <- df %>% select(Median_House_Value, everything())

# Correlation matrix
cor_matrix <- cor(df)
ggcorrplot(cor_matrix,
           method = "circle",
           type = "lower",
           lab = TRUE,
           lab_size = 3,
           colors = c("orange", "white", "seagreen"),
           title = "Correlation Matrix",
           ggtheme = theme_minimal())

# Remove unnecessary variables
df <- df %>%
  select(-Households, -Population, -Tot_Rooms)
# multicollinearity - these are double counted with the new variables and serve no purpose

## Spatial EDA

# Convert to spatial data frame
df_sf <- st_as_sf(df, coords = c("Longitude", "Latitude"), crs = 4326)
df_sf$Longitude <- st_coordinates(df_sf)[, 1]
df_sf$Latitude <- st_coordinates(df_sf)[, 2]


# Plot house prices on a map
ggplot(data = df_sf) +
  geom_sf(aes(color = Median_House_Value)) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "House Prices in California", color = "Median House Value")

# Calculate spatial weights
coords <- cbind(df$Longitude, df$Latitude)
nb <- knn2nb(knearneigh(coords, k = 4)) # warning messages, but no time to resolve and they don't invalidate the test
lww <- nb2listw(nb, style = "W")


# Calculate Moran's I
moran_test <- moran.test(df$Median_House_Value, lww)
print(moran_test)

## Prepare data for modelling 
## Split data

set.seed(451)
split <- initial_split(df, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)


## 02 Linear regression ####

# Linear model house price and income
linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

income_recipe <- recipe(Median_House_Value ~ Median_Income, data = train_data)

income_workflow <- workflow() %>%
  add_model(linear_reg_spec) %>%
  add_recipe(income_recipe)

income_fit <- fit(income_workflow, data = train_data)

# Extract the fitted model
fitted_model <- extract_fit_parsnip(income_fit)

# Print the summary of the fitted model
summary(fitted_model$fit)


ggplot(df, aes(x = Median_Income, y = Median_House_Value)) +
  geom_point() +
  geom_smooth(method = 'lm', col = 'orangered') +
  labs(title = 'Relationship between house value and income',
       x = 'Median Income',
       y = 'Median House Value')

linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

coast_recipe <- recipe(Median_House_Value ~ Distance_to_coast, data = train_data)

coast_workflow <- workflow() %>%
  add_model(linear_reg_spec) %>%
  add_recipe(coast_recipe)

coast_fit <- fit(coast_workflow, data = train_data)

pitted_model <- extract_fit_parsnip(coast_fit)
summary(pitted_model$fit)

ggplot(df, aes(x = Distance_to_coast, y = Median_House_Value)) +
  geom_point() +
  geom_smooth(method = 'lm', col = 'orangered') +
  labs(title = 'Relationship between house value and distance to coast',
       x = 'Distance to Coast',
       y = 'Median House Value')


### Multiple regression

## Multiple linear regression modelling ####
## model_A
linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

mA_recipe <- recipe(Median_House_Value ~ Median_Income + Distance_to_coast + Rooms_per_House, data = train_data)

mA_workflow <- workflow() %>%
  add_model(linear_reg_spec) %>%
  add_recipe(mA_recipe)

mA_fit <- fit(mA_workflow, data = train_data)

fitted_mA <- extract_fit_parsnip(mA_fit)
summary(fitted_mA$fit)

## Multiple linear regression modelling ####
## model_B
linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

mB_recipe <- recipe(Median_House_Value ~ Median_Income + Distance_to_coast +
                      Rooms_per_House + Population_Density, data = train_data)

mB_workflow <- workflow() %>%
  add_model(linear_reg_spec) %>%
  add_recipe(mB_recipe)

mB_fit <- fit(mB_workflow, data = train_data)

fitted_mB <- extract_fit_parsnip(mB_fit)
summary(fitted_mB$fit)


## Multiple linear regression modelling ####
# model_C

# Define the linear regression specification
linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

# Create the recipe, including Longitude and Latitude
mC_recipe <- recipe(Median_House_Value ~ Median_Income + Distance_to_coast + 
                      Rooms_per_House + Population_Density + Longitude + Latitude, data = train_data)

# Create the workflow
mC_workflow <- workflow() %>%
  add_model(linear_reg_spec) %>%
  add_recipe(mC_recipe)

# Fit the model
mC_fit <- fit(mC_workflow, data = train_data)

# Extract and summarize the fitted model
fitted_mC <- extract_fit_parsnip(mC_fit)
summary(fitted_mC$fit)


## LASSO 
# Specify the lasso regression model
lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")

# Create a recipe
lasso_recipe <- recipe(Median_House_Value ~ Median_Income + Distance_to_coast +
                         Rooms_per_House + Population_Density, data = train_data)

# Create a workflow
lasso_workflow <- workflow() %>%
  add_model(lasso_spec) %>%
  add_recipe(lasso_recipe)

# Fit the model
lasso_fit <- fit(lasso_workflow, data = train_data)

# Print the Lasso model summary
lasso_summary <- lasso_fit %>%
  extract_fit_parsnip() %>%
  tidy()

print(lasso_summary)


#cv on lasso
# Define the resampling method
set.seed(451)

cv_folds <- vfold_cv(train_data, v = 5)

# Perform cross-validation
lasso_res <- fit_resamples(
  lasso_workflow,
  resamples = cv_folds,
  metrics = metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
)

# Collect metrics
collect_metrics(lasso_res)


# Define the Ridge regression model specification
ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet")

# Create the workflow
ridge_workflow <- workflow() %>%
  add_recipe(multiple_recipe) %>%
  add_model(ridge_spec)

# Perform cross-validation
set.seed(451)
ridge_res <- fit_resamples(
  ridge_workflow,
  resamples = vfold_cv(train_data, v = 5),
  metrics = metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
)

# Collect metrics
ridge_metrics <- collect_metrics(ridge_res)
ridge_metrics

# Define the Elastic Net model specification
elastic_net_spec <- linear_reg(penalty = 0.1, mixture = 0.5) %>%
  set_engine("glmnet")

# Create the workflow
elastic_net_workflow <- workflow() %>%
  add_recipe(multiple_recipe) %>%
  add_model(elastic_net_spec)

# Perform cross-validation
set.seed(451)
elastic_net_res <- fit_resamples(
  elastic_net_workflow,
  resamples = vfold_cv(train_data, v = 10),
  metrics = metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
)

# Collect metrics
elastic_net_metrics <- collect_metrics(elastic_net_res)
elastic_net_metrics


# Collect metrics for multiple regression
multiple_res <- fit_resamples(
  mC_workflow,
  resamples = vfold_cv(train_data, v = 5),
  metrics = metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
)
multiple_metrics <- collect_metrics(multiple_res)

# Collect metrics for LASSO (assuming you have already done this)
lasso_metrics <- collect_metrics(lasso_res)

# Combine all metrics into one data frame for comparison
all_metrics <- bind_rows(
  multiple_metrics %>% mutate(model = "Multiple Regression"),
  lasso_metrics %>% mutate(model = "LASSO"),
  ridge_metrics %>% mutate(model = "Ridge"),
  elastic_net_metrics %>% mutate(model = "Elastic Net")
)

# Print the combined metrics
print(all_metrics)


# Feature importance 

multiple_vip <- vip(fitted_mC) +
  ggtitle("Multiple Regression Feature Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Feature importance for LASSO
lasso_fit <- fit(lasso_workflow, data = train_data)
lasso_vip <- vip(lasso_fit) +
  ggtitle("LASSO Feature Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Feature importance for Ridge
ridge_fit <- fit(ridge_workflow, data = train_data)
ridge_vip <- vip(extract_fit_parsnip(ridge_fit)) +
  ggtitle("Ridge Feature Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Feature importance for Elastic Net
elastic_net_fit <- fit(elastic_net_workflow, data = train_data)
elastic_net_vip <- vip(extract_fit_parsnip(elastic_net_fit)) +
  ggtitle("Elastic Net Feature Importance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Plot all VIPs
print(multiple_vip)
print(lasso_vip)
print(ridge_vip)
print(elastic_net_vip)



## Random Forest ####

# Prepare the recipe
forest_recipe <- recipe(Median_House_Value ~ ., data = train_data)

# Define the model with the importance parameter
rf_model <- rand_forest(trees = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

# Create the workflow
workflow <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(rf_model)
# Define the grid of hyperparameters to tune - 2 minutes usually
grid <- grid_regular(trees(range = c(50, 200)), levels = 5)

# Perform cross-validation
set.seed(451)
cv_folds <- vfold_cv(train_data, v = 5)
tune_results <- tune_grid(
  workflow,
  resamples = cv_folds,
  grid = grid,
  metrics = metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
)

# View cross-validation results
tune_results %>%
  collect_metrics() %>%
  arrange(mean)

# Get the best number of trees
best_trees <- tune_results %>%
  select_best(metric = "rmse")


# Finalize the workflow with the best number of trees
final_workflow <- finalize_workflow(workflow, best_trees)

# Train the final model
final_fit <- final_workflow %>%
  fit(data = train_data)

# Make predictions on the test set
predictions <- final_fit %>%
  predict(test_data) %>%
  bind_cols(test_data)

# Calculate the root mean squared error
rmse <- predictions %>%
  metrics(truth = Median_House_Value, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)
print(paste("Root Mean Squared Error:", rmse))

# Get feature importances
importance <- final_fit %>%
  extract_fit_parsnip() %>%
  vip::vi()

# Plot feature importances
importance %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Feature") +
  ylab("Importance") +
  ggtitle("Feature Importances")

# Calculate additional metrics
metrics <- predictions %>%
  metrics(truth = Median_House_Value, estimate = .pred)

# Print the metrics
print(metrics)


