from clip_client import Client

client = Client('grpc://0.0.0.0:51000')

client.profile(['Hey clip please encode my text'] * 5000, request_size=5000)
