#include <getopt.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int ofi_ver = fi_version();

    // get the provider list
    struct fi_info* hints = fi_allocinfo();
    if (!hints) {
        return EXIT_FAILURE;
    }
    // specifically ask for some capabilities and modes
    hints->caps = FI_TRIGGER | FI_RMA | FI_TAGGED | FI_RMA_EVENT;
    // get the infos
    struct fi_info* prov_list;
    int ret = fi_getinfo(ofi_ver, NULL, NULL, 0ULL, hints, &prov_list);
    if (!prov_list) {
        fprintf(stderr, "impossible to find a provider with the required mode/capabilities");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    struct fi_info* c_prov = prov_list;
    while (c_prov) {
        printf("found a provider, name %s, auto-progress ? %d\n", c_prov->fabric_attr->prov_name,
               c_prov->domain_attr->data_progress == FI_PROGRESS_AUTO);
        // go to the next one
        c_prov = c_prov->next;
    }

    fi_freeinfo(hints);
    fi_freeinfo(prov_list);
    return EXIT_SUCCESS;
}
