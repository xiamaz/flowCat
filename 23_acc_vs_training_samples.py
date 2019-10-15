import pandas as pd

## fname - excel file with accuray/F1 scores of all the runs
## figname - name of the plot to save
def plot_line(fname, figname):
    data = pd.read_excel(fname)
    # change the name of the columns accordingly
    lines = data.plot.line(x = "#training samples", y = ["f1_weighted_CNN", "f1_weighted_dense", "f1_weighted_RF"], figsize = (9,7), fontsize = 12)
    lines.set_xlabel("#Training_samples")
    lines.set_ylabel("Weighted_F1_Score")
    fig = lines.get_figure()
    fig.savefig(figname)

def main():
    json_file = "acc_all.xlsx"
    figname = "F1_vs_training_size.png"
    
    plot_line(json_file, figname)
        
if __name__ == '__main__':
       main()

