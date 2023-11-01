from imports import np
from imports import plt

def plot_label_distribution(labels, title):    
    label_counts = np.bincount(labels)
    unique_labels = np.arange(len(label_counts))
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, label_counts, align='center')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(unique_labels) 
    plt.show()
