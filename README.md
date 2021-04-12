# Run gunicorn
<!-- gunicorn --bind 0.0.0.0:5007 wsgi:app --timeout 600 -->

# run docker 
'''
docker run -p 6379:6379 --name my-firsredis -d redis
'''

# use this command for maximum performance
'''
gunicorn --worker-class gevent   --workers 2   --bind 0.0.0.0:5007   service:app --worker-connections 1000
'''


### Docker setup
docker build -t server_host .

## Go to nginx
docker build -t reverse_proxy .

## Raise orchestrator behind a reverse proxy
docker-compose up --build --force-recreate --no-deps