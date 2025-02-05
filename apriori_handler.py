from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from Helper import getTheFrozenElement
from Helper import getTheDictOfRules
def apply_apriori(data, min_support=0.6, lift_threshold=1.0):
    df = pd.DataFrame(data=data)
    df = df.astype(bool)  
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(frequent_itemsets)
    dict_of_fi = {}
    unfrz_fi_support = frequent_itemsets['support'].tolist()
    frz_fi_itemsets = frequent_itemsets['itemsets'].tolist()
    unfrz_fi_itemsets = getTheFrozenElement(frz_fi_itemsets)
    dict_of_fi["support"] = unfrz_fi_support
    dict_of_fi["itemsets"] = unfrz_fi_itemsets
    print(dict_of_fi)



    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)
    print(rules)
    dict_of_rules = getTheDictOfRules(rules=rules)
    print(dict_of_rules)
    

    return dict_of_fi, dict_of_rules
