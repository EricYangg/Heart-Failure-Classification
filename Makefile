# Makefile
# author: Eric Yang
# date: 2025-12-11

.PHONY: all clean

all: reports/heart_disease_analysis.html reports/heart_disease_analysis.pdf

# Download data
data/raw/heart.csv : scripts/01_download_data.py
	python scripts/01_download_data.py \
		--url="https://epl.di.uminho.pt/~jcr/AULAS/ATP2021/datasets/heart.csv" \
		--write_to=data/raw

# Validate and split data
logs/validation_errors.log data/validated/heart_train.csv data/validated/heart_test.csv : scripts/02_validate_n_split.py \
data/raw/heart.csv
	python scripts/02_validate_n_split.py \
		--logs-to=logs \
		--raw-data=data/raw/heart.csv \
		--data-to=data/validated \
		--seed=123

# Perform EDA and preprocessing
results/figures/boxplots_numeric_features.png results/figures/numeric_dist_combined.png results/figures/categorical_dist_combined.png results/figures/correlation_heatmap_numeric_features.png data/validated/X_train.csv data/validated/y_train.csv data/validated/X_test.csv data/validated/y_test.csv : scripts/03_eda_validate.py \
data/validated/heart_train.csv \
data/validated/heart_test.csv
	python scripts/03_eda_validate.py \
		--training-data=data/validated/heart_train.csv \
		--test-data=data/validated/heart_test.csv \
		--plot-to=results/figures \
		--data-to=data/validated

# Create preprocessor
results/models/heart_preprocessor.pickle : scripts/04_preprocessor.py \
data/validated/heart_train.csv
	python scripts/04_preprocessor.py \
		--training-data=data/validated/heart_train.csv \
		--preprocessor-to=results/models \
		--seed=123

# Fit model and generate reports
results/tables/cv_results_df.csv results/tables/eval_confusion_matrix_logreg.csv results/tables/eval_classification_report_logreg.csv results/tables/fit_confusion_matrix_logreg.csv results/models/heart_lr_fit_pipeline.pickle results/figures/fit_confusion_matrix_logreg.png results/figures/eval_confusion_matrix_logreg.png : scripts/05_fit_heart_disease_model.py \
data/validated/X_train.csv \
data/validated/y_train.csv \
data/validated/X_test.csv \
data/validated/y_test.csv \
results/models/heart_preprocessor.pickle
	python scripts/05_fit_heart_disease_model.py \
		--x-train-data=data/validated/X_train.csv \
		--y-train-data=data/validated/y_train.csv \
		--x-test-data=data/validated/X_test.csv \
		--y-test-data=data/validated/y_test.csv \
		--preprocessor=results/models/heart_preprocessor.pickle \
		--pipeline-to=results/models \
		--results-to=results/tables \
		--figures-to=results/figures \
		--seed=123 \
		--cv-folds=5

# Render reports
reports/heart_disease_analysis.html reports/heart_disease_analysis.pdf : reports/heart_disease_analysis.qmd \
reports/references.bib \
results/figures/boxplots_numeric_features.png \
results/figures/categorical_dist_combined.png \
results/figures/correlation_heatmap_numeric_features.png \
results/figures/eval_confusion_matrix_logreg.png \
results/figures/fit_confusion_matrix_logreg.png \
results/figures/numeric_dist_combined.png \
results/tables/cv_results_df.csv \
results/tables/eval_classification_report_logreg.csv \
results/tables/eval_confusion_matrix_logreg.csv \
results/tables/fit_confusion_matrix_logreg.csv \
results/models/heart_lr_fit_pipeline.pickle
	quarto render reports/heart_disease_analysis.qmd --to html
	quarto render reports/heart_disease_analysis.qmd --to pdf

# Clean up analysis
clean :
	rm -rf data/raw/heart.csv
	rm -rf data/validated/*
	rm -rf logs/validation_errors.log
	rm -f results/models/heart_preprocessor.pickle \
	      results/models/heart_lr_fit_pipeline.pickle
	rm -f results/figures/boxplots_numeric_features.png \
	      results/figures/categorical_dist_combined.png \
	      results/figures/correlation_heatmap_numeric_features.png \
	      results/figures/eval_confusion_matrix_logreg.png \
	      results/figures/fit_confusion_matrix_logreg.png \
	      results/figures/numeric_dist_combined.png
	rm -f results/tables/cv_results_df.csv \
	      results/tables/eval_classification_report_logreg.csv \
	      results/tables/eval_confusion_matrix_logreg.csv \
	      results/tables/fit_confusion_matrix_logreg.csv
	rm -f reports/heart_disease_analysis.html \
	      reports/heart_disease_analysis.pdf