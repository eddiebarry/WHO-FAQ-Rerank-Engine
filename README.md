# Run gunicorn
<!-- gunicorn --bind 0.0.0.0:5007 wsgi:app --timeout 600 -->

# run docker 
'''
docker run -p 6379:6379 --name my-firsredis -d redis
'''

# use this command for maximum performance
'''
gunicorn --worker-class gevent   --workers 1   --bind 0.0.0.0:5007   patched:app
'''