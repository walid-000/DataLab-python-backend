from flask import Flask, request, jsonify
import pandas as pd
from kmeans_clustering import perform_kmeans_clustering 
from sklearn.cluster import KMeans
from apriori_handler import apply_apriori 
from flask_cors import CORS  
import numpy as np
app = Flask(__name__)
CORS(app)  


 
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


# Endpoint to perform the Elbow Method and return necessary data
@app.route('/elbow-method', methods=['POST'])
def elbow_method():
    print("elbow function started successfully")
    
    # Get data from frontend (assumed to be in JSON format)
    data = request.get_json()
    print(data)
    X = np.array(data['data'])  # data is an array of data points
    print("pandas data :", X)
    
    # Perform Elbow Method by calculating inertia for different clusters
    inertia_values = []
    for k in range(1, 11):  # Checking from 1 to 10 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    # Return the inertia values for the Elbow Method
    return jsonify({'inertia_values': inertia_values})


# Endpoint for performing KMeans clustering and returning results
@app.route('/kmeans-clustering', methods=['POST'])
def kmeans_clustering():
    # Get data from frontend
    data = request.get_json()
    print(data)
    X = np.array(data['data'])
    num_clusters = int(data['num_clusters'])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Return the clustering labels and centers for frontend plotting
    return jsonify({
        'labels': labels.tolist(),
        'centers': centers.tolist() 
    })



if __name__ == '__main__':
    app.run(debug=True)
app