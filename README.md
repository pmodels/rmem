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

We use a handmade `Makefile` to compile.
To handle the different systems, the file `make_arch/default.mak` contains the different variable definitions needed to find the dependencies etc.
Specifically we rely on the following variables:

- `CC` gives the compiler to use
- `PMI_DIR` the root directory of `pmi`
- `OFI_DIR` the root directory of `ofi`
- `OPTS` (optional) contains flags to be passed to the compilers for more flexibility. E.g. `-fsanitize=address`, `-flto` etc

The `Makefile` offers various targets by defaults:

- `rmem`: builds `rmem`
- `info`: display info about the build
- `default`: displays the info and build `rmem`
- `fast`: compiles for fast execution (equivalent to `OPTS=-O3 -DNDEBUG -flto`)
- `debug`: compiles for debug (equivalent to `OPTS=-O0 -g -fsanitize=address -fsanitize=undefined`)
- `verbose`: compiles for debug with added verbosity (equivalent to `OPTS=-DVERBOSE make debug`)
- `clean`/`reallyclean`: cleans the build


Note: if you prefer to add another make_arch file, you can also invoke it using `ARCH_FILE=make_arch/myfile make`.

### build variables

- `NO_RMA_EVENT`: disactivate the use of RMA events to track of the RMA calls on the target side, uses `writedata` based approach instead.
- `NO_WRITE_DATA`: disactivate the use of `fi_writedata` to complete the `put + signal` operation, uses a `FI_FENCE` approach instead.

## license

```
/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
```


