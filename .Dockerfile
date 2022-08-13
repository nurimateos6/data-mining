FROM python:3.8

RUN pip3 install --no-cache scikit-learn pandas urllib io

#COPY 'data/load_data' /usr/bin/load

#RUN chmod 755 /usr/bin/load

#EXPOSE 8080