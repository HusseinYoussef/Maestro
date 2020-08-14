''' This module for doing multi-processing using threads '''
from queue import Queue
from threading import Thread, Lock
import time
import random

class Pool():

        used = 0

        def __init__(self, max_threads = 5):
            self.max_threads = max(max_threads, 1)
            self.max_threads = min(max_threads, 10)            
            self.mutex = Lock()
        
        def change_used(self, val):
            self.mutex.acquire()
            self.used += val
            self.mutex.release()

        def run(self, queue, idx, function, inp):
            queue.put((idx, function(*inp)))
            self.change_used(-1)

        def go(self, function, arr):
            '''
            go is the function which should be used by the user.
            function: the function which will be invoked by each thread.
            arr: the inputs for the function. MUST BE IN THIS FORMULA [(...,), (...,), ...]
            '''
            queue = Queue() # Thread safe queue to do syncronization between threads.
            #breakpoint()
            for idx, x in enumerate(arr):
                while self.used >= self.max_threads:
                    time.sleep(random.random())
                worker = Thread(target= self.run, args= (queue, idx, function, x, ))
                worker.start()
                self.change_used(1)
                '''
                workers.append()
                if idx == len(arr)-1 or len(workers) == self.max_threads:
                    for worker in workers: worker.start()
                    for worker in workers: worker.join()
                    workers = []
                '''
            while self.used > 0:
                time.sleep(random.random())
            outputs = [None] * len(arr)
            for i in range(len(arr)):
                idx, out = queue.get()
                outputs[idx] = out
            return outputs