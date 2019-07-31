#! /usr/bin/env python

# Python's bundled WSGI server
from wsgiref.simple_server import make_server
import json
import time


new_pos = []
new_constraint = []


def application (environ, start_response):
    if environ['REQUEST_METHOD'] == 'POST':
        # the environment variable CONTENT_LENGTH may be empty or missing
        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0
        request_body = environ['wsgi.input'].read(request_body_size)
        data = json.loads(request_body)
        type = data.pop('type', None)

        if type == 'new_positions':

            new_pos.insert(0,data)
            if len(new_constraint) > 0:
                request_body = bytes('new_constraint', 'utf-8')

        elif type == 'new_constraint':
            new_constraint.insert(0,data)

        #new_pos.insert(0,data)
        status = '200 OK'
        response_headers = [
            ('Content-Type', 'text/json'),
            ('Access-Control-Allow-Origin', '*'),
            ('Content-Length', str(len(request_body)))
        ]

        start_response(status, response_headers)

        return [request_body]

    elif environ['REQUEST_METHOD'] == 'GET':

        error = ''

        if environ['QUERY_STRING'] == 'js':
            try:
                response_content = new_pos.pop()
                response_body = json.dumps(response_content)

                error = 'Update positions'
            except:
                print ("Something went wrong getting updated positions!")

        elif environ['QUERY_STRING'] == 'constraint':
            try:
                response_content = new_constraint.pop()

                response_body = json.dumps(response_content)

                error = 'Sending constraint'
            except:
                print ("Something went wrong sending new constraint")

        try:
            status = '200 OK'

            response_headers = [
                ('Content-Type', 'text/json'),
                ('Access-Control-Allow-Origin', '*'),
                ('Content-Length', str(len(response_body)))
            ]
            start_response(status, response_headers)
            response_body = bytes(response_body, 'utf-8')
            return [response_body]
        except:
            print("Error in GET request. Error in: " + error)


    elif environ['REQUEST_METHOD'] == 'OPTIONS':

        status = '200 OK'
        response_headers = [
            ('Content-Type', 'text/json'),
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Headers', '*'),
            ('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE, OPTIONS')
        ]

        start_response(status, response_headers)

        return ''

# Instantiate the server
httpd = make_server (
    '127.0.0.1', # The host name
    8051, # A port number where to wait for the request
    application # The application object name, in this case a function
)

# Wait for a single request, serve it and quit
httpd.serve_forever()

