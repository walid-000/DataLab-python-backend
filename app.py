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

@app.route('/apriori', methods=['POST'])
def apply_apriori_route():
    data = request.get_json()

    # data = reqdata['data']

    df = pd.DataFrame(data)

    frequent_itemsets, rules = apply_apriori(df)

    result = {
        'frequent_itemsets': frequent_itemsets.to_dict(orient='records'),
        'association_rules': rules.to_dict(orient='records')
    }

    return jsonify(result)

@app.route("/" , methods=['GET'])
def root():
    result = {
        "message" : "Hello World !"
    }

    return jsonify(result)


@app.route("/hello" , methods=['POST'])
def test():
    try :
        reqdata = request.get_json()
        data = reqdata['data']
        print(data)
        df = pd.DataFrame(data)

        # Apply apriori algorithm
        frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        # Return the result as JSON
        result = rules.to_json(orient="records")
        print(result)
        return jsonify(result)

    except Exception as e:
        print("error :" , str(e))
        return jsonify({"error": str(e)})

 

if __name__ == '__main__':
    app.run(debug=True)
