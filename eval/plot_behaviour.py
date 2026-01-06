import json
import matplotlib.pyplot as plt
import numpy as np

model_mapping = {
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B": "(Guider) R1-Distill-1.5B",
    "Qwen_Qwen2.5-32B": "(Target) Qwen2.5-32B",
    "dexperts-S1.5B-L32B/warmup": "Target + ThinkLogit",
    "dexperts-dpo-lora-S1.5B-L32B/warmup": "Target + ThinkLogit-DPO"
}

behaviors = ['branching', 'backtracking', 'self_verification']
behavior_names = ['Branching out', 'Backtracking', 'Self-verification']

subsets = {
    'math': ['aime2024', 'aime2025', 'amc23', 'MATHL5'],
    'aime2024': ['aime2024'],
    'aime2025': ['aime2025'],
    'amc23': ['amc23'],
    'MATHL5': ['MATHL5'],
    'science': ['gpqa'],
    'coding': ['lcb_200'],
    'all': ['aime2024', 'aime2025', 'amc23', 'MATHL5', 'gpqa', 'lcb_200']
}

def main(mode='norm'):
    # Load counts from file
    with open('behaviour_analysis_counts.json', 'r') as f:
        counts_data = json.load(f)
    
    # Reorder to show Target first, then Guider
    desired_order = [
        "Qwen_Qwen2.5-32B",
        "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B",
        "dexperts-S1.5B-L32B/warmup",
        "dexperts-dpo-lora-S1.5B-L32B/warmup"
    ]
    colors = ['#3070ad', '#469c76', '#c66526', '#c07da5']
    
    # Increase font size for all elements
    plt.rcParams.update({'font.size': 26})
    
    # Create 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    subsets_to_plot = ['math', 'science', 'coding']
    
    for idx, subset in enumerate(subsets_to_plot):
        ax = axes[idx]
        datasets = subsets[subset]
        
        # Compute averages for the subset
        avgs = {}
        models = list(model_mapping.keys())
        for model in models:
            avgs[model] = {}
            for behavior in behaviors:
                all_vals = []
                for dataset in datasets:
                    all_vals.extend(counts_data[mode][dataset][model][behavior])
                avgs[model][behavior] = sum(all_vals) / len(all_vals)
        
        models = [m for m in desired_order if m in avgs]
        
        # Plot
        x = np.arange(len(behaviors))
        width = 0.15
        for i, model in enumerate(models):
            if mode == 'norm':
                values = [avgs[model][behavior] * 1000 for behavior in behaviors]
            else:
                values = [avgs[model][behavior] for behavior in behaviors]
            ax.bar(x + i*width, values, width, label=model_mapping.get(model, model), color=colors[i], edgecolor='black')
        
        ax.set_title(f'{subset.capitalize()}')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(behavior_names)
        if idx == 0:
            if mode == 'norm':
                ax.set_ylabel('Frequency (‰)')
            else:
                ax.set_ylabel('Average Count')
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=22)
    
    plt.tight_layout()
    outfile = f'figs/reasoning_behavior_analysis_combined_{mode}.pdf'
    print(f'Saving figure to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'raw'
    main(mode)