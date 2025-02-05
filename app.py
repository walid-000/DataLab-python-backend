from flask import Flask, request, jsonify
import pandas as pd
from kmeans_clustering import perform_kmeans_clustering 
from apriori_handler import apply_apriori 
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  


@app.route('/cluster', methods=['POST'])
def cluster_data():

    data = request.get_json()
    
    points = data['points']
    n_clusters = data['n_clusters']

    result = perform_kmeans_clustering(points, n_clusters)
    return jsonify(result)

 
@app.route('/apriori' , methods=['POST'])
def apriori_route(): 
    reqdata = request.get_json()
    data = reqdata['data']
    print("data" , data)
    frequent_itemsets, rules = apply_apriori(data, min_support=0.5, lift_threshold=1.0)
    print(rules)
    print(frequent_itemsets)
    

    
    # Convert the results to dictionaries for JSON response
    result = {
        'frequent_itemsets': frequent_itemsets ,
        'association_rules': rules
    }

    # Return the result as a JSON response
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
app