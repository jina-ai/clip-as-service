import sys

try:
    import gpu_env
except:
    print('no GPUutils!')
from utils.server import ServerTask

if __name__ == '__main__':
    server = ServerTask(sys.argv[1], max_seq_len=50)
    server.start()
    server.join()
