# ---------------------------
# 0) Set Working Directory
# ---------------------------
setwd("C:/Users/josep/OneDrive/سطح المكتب/BYU/Fall 25/Stat 348/AmazonEmployeeAccess")

# ---------------------------
# 1) Libraries
# ---------------------------
library(tidymodels)
library(embed)   # target encoding
library(vroom)

# ---------------------------
# 2) Read Data
# ---------------------------
train <- vroom("train.csv", show_col_types = FALSE)
test  <- vroom("test.csv",  show_col_types = FALSE)
sampleSub <- vroom("sampleSubmission.csv", show_col_types = FALSE)

# Outcome must be a factor; make "1" the event level for AUC
train <- train %>%
  mutate(ACTION = factor(ACTION, levels = c(1, 0)))

# ---------------------------
# 3) Recipe (categoricals → rare-level collapse → target encoding)
# ---------------------------
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = factor) %>%             # convert predictors to factors
  step_other(all_nominal_predictors(), threshold = 0.001) %>%   # combine rare levels
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))  # target encode

# ---------------------------
# 4) Penalized Logistic (glmnet)
# ---------------------------
my_mod <- logistic_reg(
  penalty = tune(),   # λ
  mixture = tune()    # ν (0=ridge, 1=lasso, (0,1)=elastic net)
) %>% 
  set_engine("glmnet")

# ---------------------------
# 5) Workflow
# ---------------------------
amazon_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

# ---------------------------
# 6) Light CV (fast)
# ---------------------------
set.seed(123)
folds <- vfold_cv(train, v = 5)

# ---------------------------
# 7) Small tuning grid (fast)
# ---------------------------
tuning_grid <- grid_regular(
  penalty(),  # default range on log scale
  mixture(),  # 0..1
  levels = 3  # 9 total combos → quick
)

# ---------------------------
# 8) Tune with AUC only (avoid precision/recall warnings)
# ---------------------------
CV_results <- amazon_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

# ---------------------------
# 9) Pick best by AUC & show it
# ---------------------------
print(show_best(CV_results, metric = "roc_auc", n = 1))
best_params <- select_best(CV_results, metric = "roc_auc")

# ---------------------------
# 10) Finalize & fit on all training data
# ---------------------------
final_wf <- amazon_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

# ---------------------------
# 11) Predict probabilities on test and build submission
# ---------------------------
final_predictions <- predict(final_wf, new_data = test, type = "prob")

submission <- bind_cols(
  sampleSub["id"],
  final_predictions[".pred_1"]  # Pr(ACTION=1)
) %>%
  rename(ACTION = .pred_1)

# ---------------------------
# 12) Write CSV (Kaggle-ready)
# ---------------------------
vroom_write(submission, "mySubmission.csv")

# Optional quick glance at CV metrics table
collect_metrics(CV_results)

