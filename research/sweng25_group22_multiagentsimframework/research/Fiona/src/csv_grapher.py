import csv
import matplotlib.pyplot as plt 

def graph_csv_1col_freq(path: str):
    map = {};
    with open(path,'r') as csvfile: 
        plots = csv.reader(csvfile, delimiter = ',');
        
        next(plots)
        for row in plots: 
            if row[0] in map:
                map[row[0]] += 1;
            else:
                map[row[0]] = 1;
    
    plt.bar(map.keys(), map.values(), color = 'g', width = 0.72, label = "Frequency");
    plt.xlabel('Winners');
    plt.ylabel('Frequency');
    plt.title('Frequency of each winner');
    plt.legend();
    plt.show();