from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def save_confusion_matrix(labels, pred):
    cf_matrix = confusion_matrix(labels, pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="g")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    plt.savefig("confusion.png")
