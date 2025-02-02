from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def apply_apriori(data, min_support=0.6, lift_threshold=1.0):

    # Step 1: Apply Apriori to find frequent itemsets
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    # Step 2: Generate association rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)

    return frequent_itemsets, rules
