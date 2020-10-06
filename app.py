import flask
import sys, json
sys.path.append("./models")
from flask import request, jsonify
from T5Reranker import T5Reranker

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def hello_world():
    return 'Hello, World! The reranking service is up :)'

@app.route('/api/v1/reranking', methods=['GET'])
def rerank_documents():
    """
    This api reranks user queries and search result documents

    Inputs
    ------
    Expects a api call of the form : ""
    
    Query : String
        The string from which we need to find the most relevant 
        search result for

    Outputs
    -------
    Json Object : 
        The form of the json object is as follows : -
    """
    global T5Reranker

    params = json.loads(request.json)
    # If first time being sent, calculate a unique id
    query = params['query']
    texts = params['texts']
    
    scoreDocs = T5Reranker.rerank(query,texts)

    response = {
        'scoreDocs' : scoreDocs,
    }

    # Logging
    original_stdout = sys.stdout 
    with open('rerank_log.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('$'*80)
        print("The user query is ", query)
        print("The documents to rerank are ", texts)
        print("The results of the reranking is ", scoreDocs)
        print('$'*80)
        sys.stdout = original_stdout

    return jsonify(response)


if __name__ == '__main__':
    T5Reranker = T5Reranker()
    app.run(host='0.0.0.0', port = 5007)