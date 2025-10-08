# =======================================================
# Amazon Employee Access - Wrangling Categorical Data HW
# =======================================================

# ---------------------------
# 0) Set Working Directory
# ---------------------------
setwd("C:/Users/josep/OneDrive/سطح المكتب/BYU/Fall 25/Stat 348/AmazonEmployeeAccess")

# ---------------------------
# 1) Libraries
# ---------------------------
library(vroom)
library(dplyr)
library(ggplot2)
library(ggmosaic)
library(tidymodels)

# ---------------------------
# 2) Read Data
# ---------------------------
train <- vroom("train.csv", show_col_types = FALSE)
test  <- vroom("test.csv",  show_col_types = FALSE)
sampleSub <- vroom("sampleSubmission.csv", show_col_types = FALSE)

# ---------------------------
# 3) Basic EDA (fixed version, based on PowerPoint logic)
# ---------------------------

# Check class distribution
train %>%
  count(ACTION) %>%
  mutate(prop = n / sum(n))

# (A) RESOURCE vs ACTION
ggplot(train, aes(x = factor(RESOURCE), fill = factor(ACTION))) +
  geom_bar(position = "fill") +
  labs(
    title = "RESOURCE vs ACTION (Proportion of ACTION per RESOURCE)",
    x = "RESOURCE",
    y = "Proportion",
    fill = "ACTION"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank())

# (B) ROLE_DEPTNAME vs ACTION
ggplot(train, aes(x = factor(ROLE_DEPTNAME), fill = factor(ACTION))) +
  geom_bar(position = "fill") +
  labs(
    title = "ROLE_DEPTNAME vs ACTION (Proportion of ACTION per Department)",
    x = "ROLE_DEPTNAME",
    y = "Proportion",
    fill = "ACTION"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank())

# ---------------------------
# 4) Recipe for Categorical Wrangling (as shown in slides)
# ---------------------------

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%             # make all predictors categorical
  step_other(all_nominal_predictors(), threshold = 0.001) %>%   # combine rare (<0.1%) categories into "other"
  step_dummy(all_nominal_predictors())                          # convert categorical vars into dummy columns

# ---------------------------
# 5) Prep and Bake
# ---------------------------
prepped <- prep(amazon_recipe)
baked_train <- bake(prepped, new_data = train)
baked_test  <- bake(prepped, new_data = test)

# ---------------------------
# 6) Output Results
# ---------------------------
cat("✅ Number of columns in baked training data:", ncol(baked_train), "\n")
print(dim(baked_train))
print(head(baked_train, 3))

# ---------------------------
# 7) Optional: Save Processed Data
# ---------------------------
# vroom_write(baked_train, "baked_train.csv", delim = ",")
# vroom_write(baked_test,  "baked_test.csv",  delim = ",")
