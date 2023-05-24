# `RMEM` - RMA-accessible memory for MPI

## Dependencies

### PMI + hydra

PMI (client) and hydra (server) provide process management abilities.
Download them from the [mpich website](https://www.mpich.org/downloads/).
To build `rmem`, only the `PMI` library is needed, although hydra (or any other `PMI`) server is needed to execute `rmem`.

Here are some examples of usefull commands with `hydra`:

```bash
# run 2 processes, 1 process-per-node, label the output
mpiexec -n 2 -ppn 1 -l
# run 2 processes, bind each of them to a core
mpiexec -bind-to core
```

### OFI - `libfabric`

Libfabric (aka `ofi`) comes with few providers:
- open-source: `sockets`,`psm3` (can be built on top of `ib-verbs` for IB networks)
- closed-source: `cxi`


To build `libfabric`, there are different options (ex: `brew install libfabric` on macos), here is how to do it from source:
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


## build `rmem`

```bash
./autogen.sh
CC=$(CC) CXX=$(CXX) ./configure --enable-fast --with-ofi=<path_to_ofi> --with-pmi=<path-to-pmi>
make -j 8
```


## configure options

- `--enable-verbose`: enable the `m_verb` macros, leading to a more verbose log
- `--enable-fast`: enable a few optimizations including `-flto` and `-DNDEBUG`. WARNING: conflicts with `--enable-asan`
- `--enable-asan`: enable the address sanitizer, on macos use `ASAN_OPTIONS=detect_leaks=1` to get a report on the memory leaks. WARNING: conflicts with `--enable-fast`
- `--with-ofi=<path>` set the path for `libfabric`
- `--with-pmi=<path>` set the path for `PMI`


## license

```
/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
```


