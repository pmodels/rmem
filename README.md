# `RMEM` - RMA-based protocols for GPU-to-GPU communications


Explore the potential of RMA-based APIs using `libfabric`.
Please cite out paper (see bellow).

## Dependencies

### PMI + hydra

PMI (client) and hydra (server) provide process management abilities.
Download them from the [mpich website](https://www.mpich.org/downloads/).
To build `rmem`, only the `PMI` library is needed, although hydra (or any other `PMI`) server is needed to execute `rmem`.

Here are some examples of useful commands with `hydra`:

```bash
# run 2 processes, 1 process-per-node, label the output
mpiexec -n 2 -ppn 1 -l
# run 2 processes, bind each of them to a core
mpiexec -bind-to core
```

### OFI - `libfabric`

To build `libfabric`, there are different options (ex: `brew install libfabric` on macos), here is how to do it from source:
```bash
./autogen.sh
CC=$(CC) CXX=$(CXX) ./configure --prefix=$(YOUR_PREFIX) --enable-psm3 --enable-sockets
# for CUDA support, add
--with-cuda=${CUDA_HOME} --with-gdrcopy
# for AMD support, add
--with-rocr=${ROCM_PATH}
make install -j 8
```

To build CXI (Slingshot-11 provider), you will need some workarounds:
- the main branch doesn't build on most supercomputer (lib-cxi is too old, see [here](https://github.com/ofiwg/libfabric/issues/9835)), instead use [this branch](https://github.com/thomasgillis/libfabric/tree/dev-cxi)
- install json-c with [from source](https://github.com/json-c/json-c/releases/tag/json-c-0.17-20230812) with `cmake . -DCMAKE_INSTALL_PREFIX=${HOME}/json-c` and add the option `--with-json=${HOME}/json-c`


You can check that the build is working using `fi_pingpong`:
```bash
# with mpiexec
mpiexec -n 1 ./fi_pingpong : -n 1 ./fi_pingpong localhost
# or without mpiexec
fi_pingpong & fi_pingpong localhost
```


## Build `rmem`

We use a `Makefile` to compile.
To handle the different systems, the file `make_arch/default.mak` contains the different variable definitions needed to find the dependencies etc.
Specifically we rely on the following variables:

- `CC` gives the compiler to use
- `PMI_DIR` the root directory of `pmi`
- `OFI_DIR` the root directory of `ofi`
- `OPTS` (optional) contains flags to be passed to the compilers for more flexibility. E.g. `-fsanitize=address`, `-flto` etc

The `Makefile` offers various targets by defaults:

- `rmem`: builds `rmem`
- `info`: display info about the build
- `clean`/`reallyclean`: cleans the build
- `default`: displays the info and build `rmem`
- `fast`: compiles for fast execution (equivalent to `OPTS=-O3 -DNDEBUG`)
- `debug`: compiles with debug symbols (equivalent to `OPTS=-O0 -g`)
- `verbose`: compiles for debug with added verbosity (equivalent to `OPTS=-DVERBOSE make debug`)
- `asan`: compiles with debug symbols (equivalent to `OPTS=-fsanitize=address -fsanitize=undefined make verbose`)


Note: if you prefer to add another make_arch file, you can also invoke it using `ARCH_FILE=make_arch/myfile make`.


## Runtime configuration

### Exposure epoch (ready-to-receive) (`-r`)
the ready-to-receive protocol is used to expose readiness to reception by the target to the origin of the RMA call.
- `am`: will use active messaging (`fi_send`) and pre-posted buffers at the sender
- `tag`: will use tagged messaging (`fi_tsend` and `fi_trecv`). The main performance bottleneck is unexpected messages
- `atomic`: uses an atomic operation (`fi_atomic`)


### Closure epoch (down-to-close) (`-d`)
- `am`: will use `fi_send` and pre-posted buffers at the sender
- `tag`: will use `fi_tsend` and `fi_trecv`. The main performance bottleneck is unexpected messages
- `cq_data` uses `fi_cq_data` to close the epoch, to be used with `-c order`

### Remote completion tracking (`-c`)
- `delivery` uses delivery complete (`FI_DELIVERY_COMPLETE`) on the payload operation
- `fence` uses a fence to issue the down-to-close acknowledgment
- `cq_data` uses `FI_CQ_DATA` to track remote completion
- `counter` uses `FI_REMOTE_COUNTER` to track remote completion using remote counters
- `order` use network ordering, must be used with `-d cq_data`

### Usability:
Different networks have different capabilities and limitations, here is a list of the restrictions we have encountered:
- `psm3`: does not support RMA natively, emulated in software over tag messaging, see [here](https://ofiwg.github.io/libfabric/main/man/fi_psm3.7.html)
- `verbs;ofi_rxm`: poor native support of `FI_ATOMIC`
- `cxi`: doesn't support `FI_CQ_DATA` for the moment
- `sockets`: supports everything except `FI_REMOTE_COUNTER`

## Citation

To be announced

## license

```
/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *     See COPYRIGHT in top-level directory
 */
```


