import subprocess
import argparse
import socket
import socketserver
import threading

# A server which gathers IPs and ports and scatters them to all listeners
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, num_nodes, bind_and_activate=True):
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.barrier = threading.Barrier(num_nodes)
        self.connections = []

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = str(self.request.recv(1024).strip(), 'ascii')
        # gather list of IP:PORT
        self.server.connections.append(data)
        print('Job running on %s connected\r\n' % data, end='')

        # wait until all threads are accounted for before scattering
        self.server.barrier.wait()
        self.request.sendall(bytes(' '.join(self.server.connections), 'ascii'))
        print('Sent gathered hosts to %s\r\n' % data, end='')

        # server waits for all threads to finish before actually shutting down
        self.server.shutdown()


def main():
    args = get_args()
    if args.interactive:
        print('Interactive Mode enabled: Exiting this process will kill your CCC jobs.')

    # make sure you have a resolvable hostname
    # allocate a random port
    HOST, PORT = socket.getfqdn(), 0
    with ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler, num_nodes=args.num_nodes) as server:
        ip, port = server.server_address
        threads = []
        for i in range(args.num_nodes):
            threads.append(threading.Thread(target=submit_job, args=(args, HOST, port, i)));
            threads[i].start()
        print('Waiting for connections...')
        server.serve_forever()
        
    for thread in threads:
        thread.join()

def submit_job(args, host, port, node_index):
    # ps_nums equal to the half of num_nodes
    num_ps = int(args.num_nodes / 2)
    interactive = args.interactive
    n_gpu = args.n_gpu
    # if node_index<num_ps:
    #     jbsub = ['jbsub', '-queue', 'x86_1h', '-cores', '4+1', '-proj', 'distributed_tensorflow_test']
    # else:
    jbsub = ['jbsub', '-queue', 'x86_1h', '-cores', '4+'+str(n_gpu), '-proj', 'distributed_tensorflow_test']
    inter = ['-interactive'] if interactive else []
    pyjob = ['python', args.file, '--bootstrap-host', host, '--bootstrap-port', str(port), '--num_ps', str(num_ps),'--n_gpu',str(n_gpu)]
    job = subprocess.Popen(jbsub + inter + pyjob, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    i = 0
    for line in job.stdout:
        print(line.strip('\r\n'))

        
        # fix the stty settings that `bsub -interactive` destroys        
        if i < 20 and i % 3:    
            subprocess.run(['stty', 'sane'])
        i += 1

def get_args():
    parser = argparse.ArgumentParser(description='Run N individual CCC jobs using jbsub to distribute training of KB Completion. Additional parameters are passed to python job.')
    parser.add_argument('-N', '--num_nodes', type=int, help='Total number of nodes to use', required=True)
    parser.add_argument('-G', '--n_gpu', type=int, default=1,help='GPU nums of each node to be used', required=True)
    parser.add_argument('-I', '--interactive', action='store_true', help='Routes the console output of remote jobs to this terminal')
    parser.add_argument('file', metavar='app.py', help='The Python application to run.')

    return parser.parse_args()

if __name__ == '__main__':
    main()
