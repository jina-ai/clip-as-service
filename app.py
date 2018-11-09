import sys

try:
    import gpu_env
except:
    print('no GPUutils!')
from utils.server_old import ServerTask

if __name__ == '__main__':
    server = ServerTask(sys.argv[1])
    server.start()
    server.join()
