import flask, redis
import sys, json, pdb, time
sys.path.append("./models")
from flask import request, jsonify
from flask_limiter import Limiter
from flask_caching import Cache
from models.T5Reranker import T5Reranker


app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['Reranker'] = T5Reranker()
cache = Cache(app, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': 'redis://localhost:6379'})


limiter = Limiter(
    app,
    key_func=lambda : "1",
    default_limits=["100 per second"]
)

@app.route('/')
@limiter.exempt
def hello_world():
    return 'Hello, World! The reranking service is up :)'

@cache.memoize()
def get_score_docs(query, texts):
    scoreDocs = app.config['Reranker'].rerank(query,texts)
    return scoreDocs

@app.route('/api/v1/reranking', methods=['GET'])
@cache.cached()
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

    params = json.loads(request.json)
    # If first time being sent, calculate a unique id
    query = params['query']
    texts = params['texts']
    
    scoreDocs = get_score_docs(query=query, texts=texts)

    response = {
        'scoreDocs' : scoreDocs,
    }

    # Logging
    # original_stdout = sys.stdout 
    # with open('rerank_log.txt', 'a') as f:
    #     sys.stdout = f # Change the standard output to the file we created.
    # print('$'*80)
    # print("The user query is ", query)
    # print("The documents to rerank are ", texts)
    # print("The results of the reranking is ", scoreDocs)
    # print('$'*80)
    # sys.stdout = original_stdout

    return jsonify(response)


if __name__ == '__main__':
    T5Reranker = T5Reranker()
    app.run(host='0.0.0.0', port = 5007)