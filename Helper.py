required_columns = ['antecedents', 'consequents', 'antecedent support','consequent support', 'support', 'confidence', 'lift','representativity', 'leverage', 'conviction']
def getTheFrozenElement(frozen_list):
    unfrozen_list = []
    for fr in frozen_list :
        if len(fr) == 1 :
            unfrozen_list.extend(list(fr))
        elif len(fr) > 1 :
            unfrozen_list.append(list(fr))
    return unfrozen_list


def getTheDictOfRules(rules , listofColumn = required_columns) :
    dict_of_rules = {}
    for i in listofColumn :
        if i == "antecedents" or i == "consequents" :
            unfronzen_list = getTheFrozenElement(rules[i].tolist())
            print(f' {i} {unfronzen_list}')
            dict_of_rules[i] = unfronzen_list
        else :
            print(f" {i}  {rules[i].tolist()} ")
            dict_of_rules[i] = rules[i].tolist()
    return dict_of_rules
    
