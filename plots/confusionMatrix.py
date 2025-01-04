#test
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

# Custom color palette
custom_palette = sns.color_palette("Set2", 8)
DEFAULT_OUTPUT_FILENAME = "_confusionMatrix.png"
DEFAULT_OUTPUT_FILEPATH = "/Users/uw20204/Documents/figures/"
DEFAULT_CLASS_NAMES = [0, 1]
NEGATIVE_DATASET = "GnomAD"

def generate_confusion_matrix(cnf_matrix, classes, title='Confusion matrix', output_filename=DEFAULT_OUTPUT_FILENAME, output_path=DEFAULT_OUTPUT_FILEPATH):
    
    """
    Generate and visualize a confusion matrix.

    Parameters:
    - cnf_matrix (numpy.ndarray): The confusion matrix to be visualized.
    - classes (list): A list of class labels.
    - normalise (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
    - title (str, optional): The title of the confusion matrix plot. Defaults to 'Confusion matrix'.
    - output_filename (str, optional): The name of the output file to save the plot. Defaults to '_confusionMatrix.png'.

    Returns:
    - numpy.ndarray: The confusion matrix used for plotting.

    Usage:
    - generate_confusion_matrix(y_true, y_pred, classes=["class1", "class2"], normalize=True, title="Normalized Confusion Matrix", output_filename="confusion_matrix.png")
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    # Use custom color palette
    sns.set_palette(custom_palette)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Calculate the percentage values
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        percentage = format(cnf_matrix_norm[i, j], '.2%')  # Convert to percentage format
        count = format(cnf_matrix[i, j], 'd')
        plt.text(j, i, f"{count}\n({percentage})", horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False) 
    # Save the figure with a high DPI
    plt.savefig(output_path + output_filename.replace(" ", "_") + ".png", dpi=300)
    plt.show()

def plot_confusion_matrix(predicted_labels_list, y_test_list, title):
    """
    Plot non-normalised and normalised confusion matrices.

    Parameters:
    - predicted_labels_list (list or array-like): Predicted labels.
    - y_test_list (list or array-like): True labels.
    - class_names (list, optional): A list of class labels. Defaults to None.

    Usage:
    - plot_confusion_matrix(predicted_labels, true_labels, class_names=["class1", "class2"])
    """

    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=DEFAULT_CLASS_NAMES, title=title, output_filename = title)

    plt.show()

