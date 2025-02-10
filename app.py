from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from apriori_handler import apply_apriori 
from flask_cors import CORS  
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
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

@app.route('/api/one-sample-ttest', methods=['POST'])
def one_sample_ttest():
    # Get the data from the POST request
    data = request.json.get('data')
    population_mean = int(request.json.get('population_mean'))
    print(data)
    print(population_mean)
    # Ensure we have the required data
    if not data or population_mean is None:
        return jsonify({"error": "Invalid input data"}), 400
    
    # Convert data to numpy array for processing
    sample_data = np.array(data , dtype=int)

    # Perform the one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(sample_data, population_mean)

    # Return the results as JSON
    return jsonify({
        "t_statistic": t_statistic,
        "p_value": p_value
    })

@app.route('/api/two-sample-ttest', methods=['POST'])
def two_sample_ttest():
    # Get data from the request payload
    group1_data = request.json.get('group1_data')
    group2_data = request.json.get('group2_data')

    # Ensure that the required data is provided
    if not group1_data or not group2_data:
        return jsonify({"error": "Both groups are required"}), 400

    # Convert the data to numpy arrays for processing
    group1 = np.array(group1_data , dtype=int)
    group2 = np.array(group2_data , dtype=int)
    print(group1)
    print(group2)

    # Perform the two-sample t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Return the results
    return jsonify({
        "t_statistic": t_statistic,
        "p_value": p_value
    })

@app.route('/api/paired-sample-ttest', methods=['POST'])
def paired_sample_ttest():
    # Get data from the request payload
    group1_data = request.json.get('group1_data')
    group2_data = request.json.get('group2_data')

    # Ensure that the required data is provided
    if not group1_data or not group2_data:
        return jsonify({"error": "Both groups are required"}), 400

    # Convert the data to numpy arrays for processing
    group1 = np.array(group1_data , dtype=int)
    group2 = np.array(group2_data , dtype=int)

    # Perform the paired sample t-test
    t_statistic, p_value = stats.ttest_rel(group1, group2)

    # Return the results
    return jsonify({
        "t_statistic": t_statistic,
        "p_value": p_value
    })

@app.route('/api/pearson-correlation', methods=['POST'])
def pearson_correlation():

    # Get data from the request payload
    data = request.get_json()
    print(data)
    print()
    group1_data = request.json.get('group1_data')
    group2_data = request.json.get('group2_data')
    print(group1_data)
    print(group2_data)
    # Ensure that the required data is provided
    if not group1_data or not group2_data:
        return jsonify({"error": "Both groups are required"}), 400

    # Convert the data to numpy arrays for processing
    group1 = np.array(group1_data , dtype=int)
    group2 = np.array(group2_data , dtype=int)
    print(group1)
    print(group2)
    # Perform the Pearson correlation test
    correlation_coefficient, p_value = pearsonr(group1, group2)

    # Return the results
    return jsonify({
        "correlation_coefficient": correlation_coefficient,
        "p_value": p_value
    })


if __name__ == '__main__':
    app.run(debug=True)
app