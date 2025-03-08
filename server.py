# server.py
# a simple server to serve the wasm page for the planet sim
# code adapted from https://dev.to/gavi/custom-mime-type-with-python-3-httpserver-530l

import http.server
import socketserver
from urllib.parse import urlparse

# every extension that needs a mime type that isn't the default text/plain
custom_mime_types = {
    ".js": "text/javascript",
}

# a custom server that respects the mime type map defined above
class CustomMimeTypesHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        # get the file extension
        url = urlparse(path)
        file_ext = url.path
        pos = file_ext.rfind('.')
        if pos != -1:
            file_ext = file_ext[pos:]
        else:
            file_ext = ""

        # Check if the file extension has a custom MIME type
        if file_ext in custom_mime_types:
            return custom_mime_types[file_ext]

        # Fallback to the default MIME type guessing
        return super().guess_type(path)

# Set the handler to use the custom class
handler = CustomMimeTypesHTTPRequestHandler

# Set the server address and port
server_address = ("localhost", 8080)

# Create the server and bind the address and handler
httpd = socketserver.TCPServer(server_address, handler)

print(f"Serving on http://{server_address[0]}:{server_address[1]}")
httpd.serve_forever()
