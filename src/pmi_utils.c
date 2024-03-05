/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#include "pmi_utils.h"

#include <pmi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rmem_utils.h"

static int pmi_key_count = 0;

// coming from MPICH (mpir_pmi.c)
static int hex(unsigned char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    } else if (c >= 'a' && c <= 'f') {
        return 10 + c - 'a';
    } else if (c >= 'A' && c <= 'F') {
        return 10 + c - 'A';
    } else {
        return m_failure;
    }
}

static void encode(const size_t addr_len, const uint8_t* addr, const int key_len, char** key) {
    m_assert(2 * addr_len + 1 <= key_len, "addr len = %ld, converted = %ld  <= val len %d",
             addr_len, 2 * addr_len + 1, key_len);
    for (int i = 0; i < addr_len; ++i) {
        snprintf((*key) + 2 * i, key_len - 2 * i, "%02X", addr[i]);
    }
    (*key)[2 * addr_len] = '\0';
}

static void decode(const int key_len, const char* key, const size_t addr_len, uint8_t** addr) {
    // decode
    for (int j = 0; j < addr_len; ++j) {
        (*addr)[j] = (hex(key[2 * j]) << 4) + hex(key[2 * j + 1]);
    }
}

int pmi_init() {
    int is_init = 0;
    int is_spawned = 0;
    m_pmi_call(PMI_Initialized(&is_init));
    m_assert(!is_init, "PMI cannot be initialized twice");
    m_pmi_call(PMI_Init(&is_spawned));

    return m_success;
}

int pmi_get_comm_id(int* id_world, int* n_world) {
    m_pmi_call(PMI_Get_rank(id_world));
    m_pmi_call(PMI_Get_size(n_world));
    return m_success;
}

// based on the local addr value, constructs a list of addresses for all the ranks in the world
// - [in] addr_len: the size (in bytes) of an address
// - [in] addr: the local address
// - [out] addr_world: a list of the address of everybody in the world, the array must be allocated
// to n_world * addr_len size where n_world is obtained using pmi_get_comm_id
int pmi_allgather(const size_t addr_len, const void* addr, void** addr_world) {
    // get the local rank
    int rank;
    int size;
    m_pmi_call(PMI_Get_rank(&rank));
    m_pmi_call(PMI_Get_size(&size));
    m_verb("PMI world is %d large, my id = %d", size, rank);

    // get the max lengths
    int name_max_len, key_max_len, val_max_len;
    m_pmi_call(PMI_KVS_Get_name_length_max(&name_max_len));
    m_pmi_call(PMI_KVS_Get_key_length_max(&key_max_len));
    m_pmi_call(PMI_KVS_Get_value_length_max(&val_max_len));

    // open a kvsname
    char* kvs = calloc(name_max_len, sizeof(char));
    m_pmi_call(PMI_KVS_Get_my_name(kvs, name_max_len));

    //---------------------------------------------------------------------------------------------
    // store the key + value into kvsname
    char* kvs_key = calloc(key_max_len, sizeof(char));
    snprintf(kvs_key, key_max_len, "key%d-%d",++pmi_key_count, rank);

    // convert the bin value to a string
    // copy into the string. every byte (uint8_t) will be converted in hexadecimal using 2 caracters
    // (2 bytes) make sure we fit into the string + '\0' at the end of it
    char* kvs_val = calloc(val_max_len, sizeof(char));
    encode(addr_len,(uint8_t*)addr,val_max_len,&kvs_val);

    //---------------------------------------------------------------------------------------------
    // exchange with others
    m_pmi_call(PMI_KVS_Put(kvs, kvs_key, kvs_val));
    m_pmi_call(PMI_KVS_Commit(kvs));
    m_pmi_call(PMI_Barrier());

    //---------------------------------------------------------------------------------------------
    // obtain the values of others
    // uint8_t* addr_list = (uint8_t*)calloc(size * addr_len, sizeof(char));
    uint8_t* addr_list = *addr_world;
    // copy my own value
    memcpy(addr_list + rank * addr_len, addr, addr_len);
    // get the value of others
    for (int i = 0; i < size; ++i) {
        // do no copy myfself
        if (i == rank) {
            continue;
        }
        // get the other's key
        int val_len = val_max_len;
        snprintf(kvs_key, key_max_len, "key%d-%d", pmi_key_count, i);
        memset(kvs_val, 0, val_max_len);
        m_verb("KVS_GET(%s)",kvs_key);
        m_pmi_call(PMI_KVS_Get(kvs, kvs_key, kvs_val, val_len));

        uint8_t* c_addr = addr_list + i * addr_len;
        decode(val_max_len, kvs_val, addr_len,&c_addr);

    }

    //---------------------------------------------------------------------------------------------
    // cleanup and return
    free(kvs_val);
    free(kvs_key);
    free(kvs);

    return m_success;
}

int pmi_finalize() {
    m_pmi_call(PMI_Finalize());
    return m_success;
}
