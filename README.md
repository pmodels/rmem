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
- `clean`/`reallyclean`: cleans the build
- `default`: displays the info and build `rmem`
- `fast`: compiles for fast execution (equivalent to `OPTS=-O3 -DNDEBUG`)
- `debug`: compiles with debug symbols (equivalent to `OPTS=-O0 -g`)
- `verbose`: compiles for debug with added verbosity (equivalent to `OPTS=-DVERBOSE make debug`)
- `asan`: compiles with debug symbols (equivalent to `OPTS=-fsanitize=address -fsanitize=undefined make debug`)


Note: if you prefer to add another make_arch file, you can also invoke it using `ARCH_FILE=make_arch/myfile make`.


## possible modes

### ready-to-receive (`-r`)
the ready-to-receive protocol is used to expose readiness to reception by the target to the origin of the RMA call.
- `am`: will use `fi_send` and pre-posted buffers at the sender
- `tag`: will use `fi_tsend` and `fi_trecv`. The main performance bottleneck is unexpected messages
- `atomic`: uses an atomic (:warning: currently broken?)


### down-to-close (`-d`)
- `am`: will use `fi_send` and pre-posted buffers at the sender
- `tag`: will use `fi_tsend` and `fi_trecv`. The main performance bottleneck is unexpected messages

### remote completion (`-c`)
- `delivery complete` uses `FI_DELIVERY_COMPLETE` on the payload
- `fence` uses a fence to issue the down-to-close acknowledgment
- `cq_data` uses `FI_CQ_DATA` to track remote completion
- `counter` uses `FI_REMOTE_COUNTER` to track remote completion using remote counters

### Usability:
Different networks have different capabilities, here is a list of them (v1.19):
- `psm3`: does not support RMA natively
- `verbs;ofi_rxm`: supports `FI_MSG`, `FI_TAGGED`, `FI_DELIVERY_COMPLETE`, `FI_CQ_DATA`, `FI_ATOMIC` (not natively)
- `cxi`: supports everything except `FI_CQ_DATA` (`FI_MSG`, `FI_TAGGED`, `FI_ATOMIC`, `FI_DELIVERY_COMPLETE`, `FI_REMOTE_COUNTER`, `FI_FENCE`)
- `sockets`: supports everything except `FI_REMOTE_COUNTER` (unsure though it doesn't support it)


### performance consideration
- when used with GPU, avoid the use of `FI_DELIVERY_COMPLETE`, it requires a `StreamSynchronize`. To avoid this, `fi_recv` can use `FI_TRANSMIT_COMPLETE` and the user is responsible to `StreamSynchronize` manually (which stream? no idea).


## license

```
/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
```


