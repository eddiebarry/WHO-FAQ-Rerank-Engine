FROM python:3.7

RUN pip install -r requrements.txt

EXPOSE 5009

# Tuned for performance on 16 core system
# change workers from 17 to 2*NUM_CPU + 1
# ENTRYPOINT [ "gunicorn", "--worker-class", "gevent", "--bind", "0.0.0.0:5009", "wsgi:app", '--workers', "17", "--worker-connections", "2000", "--timeout", "60", "--preload"]

ENTRYPOINT gunicorn --worker-class gevent --bind 0.0.0.0:5009   wsgi:app --workers 2 --worker-connections 2000 --timeout 60 --preload
# gunicorn --bind 0.0.0.0:5009 wsgi:app --timeout 600 --workers=9
# local test
# gunicorn --bind 0.0.0.0:5009 wsgi:app --timeout 100 