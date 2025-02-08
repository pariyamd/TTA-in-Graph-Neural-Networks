from collections import defaultdict
from main_function import main
import numpy as np

if __name__ == '__main__':
    source_domain = "DBLP"
    target_domain = "ACM"
    metric = "macro"  # macro micro 
    params = {"source_dataset": source_domain, "target_dataset": target_domain, "metric": metric}
        
    seeds = [1, 3, 5, 7, 9] # 1, 3, 5, 7, 9
    
    score_list = []
    micros=[]
    macros=[]
    for random_seed in seeds:
        params["random_seed"] = random_seed
        micro,macro = main(params)   
        micros.append(micro)
        macros.append(macro)
    print(micros)
    print(macros)
    # print("{}: {}_{}:\tscore\t{:.4f}\tstd\t{:.4f}".format(params['metric'], params["source_dataset"], params["target_dataset"], np.mean(score_result), np.std(score_result)))
   