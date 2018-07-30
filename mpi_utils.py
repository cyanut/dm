from mpi4py import MPI
import numpy as np
from utils import frame_hash

class MPI_IPool():
    def __init__(self, comm, sources, buf_size=2**20):
        self.comm = comm
        self.sources = sources
        self.pending_reqs = {}
        self.buf = np.zeros(buf_size)
        self.buf_size = buf_size

    def irecv_all(self, source, tag=MPI.ANY_TAG):
        more = True
        res = []
        if not (source, tag) in self.pending_reqs:
            buf = np.empty_like(self.buf)
            self.pending_reqs[(source, tag)] = \
                (self.comm.irecv(buf=buf, source=source, tag=tag), buf)

        req, _ = self.pending_reqs[(source, tag)]
        while more:
            is_data, data = req.test()
            if data is not None:
                res.append(data)
                buf = np.empty_like(self.buf)
                self.pending_reqs[(source, tag)] = \
                    (self.comm.irecv(buf=buf, source=source, tag=tag), buf)
            else:
                more = False
        return res

    def Irecv_all(self, sample_buf, source, tag=MPI.ANY_TAG):
        more = True
        res = []
        if not (source, tag) in self.pending_reqs:
            buf = np.empty_like(sample_buf)
            self.pending_reqs[(source, tag)] = \
                (self.comm.Irecv(buf, source=source, tag=tag), buf)
        while more:
            req, buf = self.pending_reqs[(source, tag)]
            if req.Test():
                res.append(buf)
                buf = np.empty_like(sample_buf)
                self.pending_reqs[(source, tag)] = \
                    (self.comm.Irecv(buf, source=source, tag=tag), buf)
            else:
                more = False
        print("Irecv_all: source={}, frames={}".format(source, [frame_hash(x) for x in res]))
        return res

    def icollect_data(self, tag):
        return self._filter_empty([self.irecv_all(source=i, tag=tag) for i in self.sources], flatten=True)

    def Icollect_data(self, buf, tag):
        res, idx = self._filter_empty([self.Irecv_all(source=i, tag=tag, sample_buf=buf) for i in self.sources], flatten=True)
        return (np.array(res), np.array(idx, dtype=int))

    def iscatter_data(self,  data, tag, idx=None):
        if idx is None:
            idx = self.sources
        for i, d in zip(idx, data):
            self.comm.isend(d, dest=i, tag=tag)

    def Iscatter_data(self, data, tag, idx=None):
        if idx is None:
            idx = self.sources
        for k, i in enumerate(idx):
            self.comm.Isend(data[k], dest=i, tag=tag)
            
    def ibcast_data(self, data, tag, idx=None):
        if idx is None:
            idx = self.sources
        for i in idx:
            self.comm.isend(data, dest=i, tag=tag)
            
    def Ibcast_data(self, data, tag, idx=None):
        if idx is None:
            idx = self.sources
        for i in idx:
            self.comm.Isend(data, dest=i, tag=tag)

    def _filter_empty(self, arr, flatten):
        res = []
        idx = []
        for i, d in zip(self.sources, arr):
            res += d
            idx += [i] * len(d)
        return (res, idx)


if __name__ == "__main__":
    import numpy as np
    import time
    mp = MPI.COMM_WORLD
    rank = mp.Get_rank()
    mp_size = mp.Get_size()

    if rank >= 1:
        print(rank)
        for i in range(4):
            for j in range(10):
                mp.Isend(np.arange(10)+rank+j*10., dest=0, tag=5)
                mp.Isend(np.arange(10)+rank+j*10., dest=0, tag=5)
            mp.Isend(np.arange(10)+rank, dest=0, tag=6)
            mp.isend("hello from rank {}".format(rank), dest=0, tag=3)
            mp.isend([np.arange(50000), 5, np.arange(6)], dest=0, tag=3)
            time.sleep(5)
            print("another round from rank {}".format(rank))
        print("rank {} finished".format(rank))
        print(mp.recv(source=0, tag=7))

    elif rank == 0:
        print(rank)
        data = np.empty(10).astype(int)
        pool = MPI_IPool(comm = mp, sources=list(range(1, mp_size)))
        pool.iscatter_data(data=["you are all done 1", 
                               "you are all done 2",
                               "you are all done 3"], tag=7)
        for i in range(200):
            tag5, idx = pool.Icollect_data(np.empty(10, dtype=float), tag=5)
            if tag5.size > 0:
                print("Icollect:", tag5, idx)
            tag3, idx = pool.icollect_data(tag=3)
            if tag3:
                print("icollect:", tag3, idx)
            time.sleep(.5)
        print("finished")
            
           

