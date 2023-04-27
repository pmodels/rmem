# Triggering for MPI communications


## Dependencies

### PMI

Download from the [mpich website](https://www.mpich.org/downloads/), untar and build where desired.
The `PMI` library allow us to be compatible with `mpiexec` and run on clusters.

### OFI - Libfabric

Libfabric (`ofi`) offers the trigger operation for a few providers:
- open-source: `sockets`,`psm3` (can be built on top of `ib-verbs` for IB networks)
- closed-source: `cxi`, `gni`


To build libfabric, there are different options (`brew install libfabric`), here is how to do it from source:
```bash
./autogen.sh
CC=$(CC) CXX=$(CXX) ./configure --prefix=$(YOUR_PREFIX) --enable-psm3 --enable-sockets
make install -j 8
```

You can check that the build is working using `fi_pingpong`:
```bash
# with mpiexec
mpiexec -n 1 ./fi_pingpong : -n 1 ./fi_pingpong localhost
# or without mpiexec
fi_pingpong & fi_pingpong localhost
```


## build `trigr`

```bash
./autogen.sh
CC=$(CC) CXX=$(CXX) ./configure --with-ofi=<path_to_ofi>
make -j 8
```
