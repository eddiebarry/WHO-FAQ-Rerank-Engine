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
app.config['cache'] = Cache(app, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': os.getenv("REDIS_URL"),'CACHE_DEFAULT_TIMEOUT':3600})
app.config['cache'].clear()

limiter = Limiter(
    app,
    key_func=lambda : "1",
    default_limits=["10 per second"]
)

@app.route('/')
@limiter.exempt
def hello_world():
    return 'Hello, World! The reranking service is up :)'

@app.route('/api/v1/reranking-cache', methods=['GET'])
@limiter.limit("24000/minute;400/second")
def cache_documents():
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
    query = params['query']
    
    scoreDocs = app.config['cache'].get(query)

    if scoreDocs is None:
        return 'not in cache', 210
    
    response = {
        'scoreDocs' : scoreDocs,
    }   
    return jsonify(response)

@app.route('/api/v1/reranking', methods=['GET'])
@limiter.limit("36000/hour;600/minute;10/second")
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
    
    scoreDocs = app.config['cache'].get(query)
    if scoreDocs is None:
        scoreDocs = app.config['Reranker'].rerank(query,texts)
        app.config['cache'].set(query,scoreDocs)


    response = {
        'scoreDocs' : scoreDocs,
    }

    return jsonify(response)


if __name__ == '__main__':
    T5Reranker = T5Reranker()
    app.run(host='0.0.0.0', port = 5007)