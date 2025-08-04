from fine_tuning.plot_results import plot_ir_metric
EVALUATION_CSV_PATH = './models/ModernBERT-100k-cos/eval/Information-Retrieval_evaluation_dev_results.csv'
OUTPUT_PLOT_PATH = 'training_mrr_curve_clinical.png'
if __name__ == "__main__":
    plot_ir_metric(EVALUATION_CSV_PATH, OUTPUT_PLOT_PATH)