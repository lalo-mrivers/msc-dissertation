import pandas as pd
import matplotlib.pyplot as plt


EVALUATION_CSV_PATH = './models/ModernBERT-base-InfoNCE-fine-tuned-cos/eval/Information-Retrieval_evaluation_dev_results.csv'
OUTPUT_PLOT_PATH = 'training_mrr_curve.png'

def plot_ir_metric(csv_path: str, plot_path: str):
    """
    Reads the IR evaluation CSV and plots a key metric.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{csv_path}'")
        return

    #  METRIC
    metric_to_plot = 'cosine-MRR@10'#'cosine-MRR@10'#'cos_sim-mrr@10'

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df[metric_to_plot], marker='o', linestyle='-', label='MRR@10 on Dev Set')
    
    plt.title('Information Retrieval Performance Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Reciprocal Rank (MRR@10)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(plot_path)
    print(f"Plot saved successfully to '{plot_path}'")
    plt.show()

