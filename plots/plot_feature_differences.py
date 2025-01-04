import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

def get_mean_abs_diff(feature, id_false_negative, test_variants):
    try:
        other_variants_mean = test_variants[test_variants["driver_stat"] == 1][feature].mean()
        incorect_variant_value = test_variants[test_variants["id"] == id_false_negative][feature].item()
        abs_diff_means = abs(other_variants_mean - incorect_variant_value)
        return feature, abs_diff_means, other_variants_mean, incorect_variant_value
    except:
        print("feature not possible")

def plot_differences(confidence_scores, features, test_variants):
    # Get the ID of the incorectly classified positive variant in the gold standard dataset
    id_false_negative = confidence_scores[(confidence_scores["actual_vals"] == 1) & (confidence_scores["predicted_vals"] != 1)]["id"].item()

    # Get the mean absolute differences between correctly classified variants and the incorectly classified variant for each feature 
    mean_res = [get_mean_abs_diff(feature, id_false_negative, test_variants) for feature in features]

    # Sort values on mean difference
    mean_res = pd.DataFrame(mean_res).sort_values(1, ascending = False)
    mean_res_filtered = mean_res[:30]

    # Rename some features for visability
    mean_res_filtered.loc[291, 0] = 'mutant_AA_Hydropathy(Naderi-Manesh_et_al.,_2001)'
    mean_res_filtered.loc[257, 0]  = 'WT_AA_The_Kerr-constant(Khanarian-Moore,_1980)'

    # Re-sorted
    mean_res_filtered = pd.DataFrame(mean_res_filtered).sort_values(1, ascending = True)

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 8)) 

    custom_palette = sns.color_palette("Dark2", 4)  

    # Plotting each column
    bar_width = 0.35
    y_pos = range(len(mean_res_filtered))

    ax.barh(y_pos, mean_res_filtered[2], height=bar_width, color=custom_palette[2], label='Correctly-classified Positive Variants')
    ax.barh([p + bar_width for p in y_pos], mean_res_filtered[3], height=bar_width, color=custom_palette[3], label='Mis-classified Positive Variant')

    ax.set_yticks([p + bar_width / 2 for p in y_pos])
    ax.set_yticklabels(mean_res_filtered[0])

    ax.set_xlabel('Mean Values')
    ax.set_title('Feature for Correctly Classified Variants and the Mis-classified Variant')

    plt.legend()
    plt.tight_layout()  # Adjusts subplot params to give some padding

    # Save the figure
    plt.savefig('/Users/uw20204/Documents/figures/feature_values_comparison.png', dpi=300)  # High resolution for publication quality

    plt.show()