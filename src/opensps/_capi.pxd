# Declarations of the core library's C API (include/fsps.h of libopensps).
#
# This file is the only place that names C symbols or the header: when the core library's
# C API is renamed (fsps_* -> opensps_*, fsps.h -> opensps.h), only the strings below change.
# The Python names on the left stay the same.

cdef extern from "fsps.h" nogil:
    int API_VERSION_MAJOR "FSPS_API_VERSION_MAJOR"
    int API_VERSION_MINOR "FSPS_API_VERSION_MINOR"
    int API_VERSION_PATCH "FSPS_API_VERSION_PATCH"

    void get_driver_version "fsps_get_driver_version" (int *major, int *minor, int *patch)
    void get_last_error "fsps_get_last_error" (int *status, char *message, int message_len)
    void clear_error "fsps_clear_error" ()

    void ctx_create "fsps_context_create" (int *handle, int *status)
    void ctx_destroy "fsps_context_destroy" (int handle, int *status)
    void ctx_setup "fsps_context_setup" (int zin, const char *isoc, const char *spec,
                                         const char *dust, int handle, int *status)

    void ctx_set_int "fsps_context_set_int" (int handle, const char *key, int value, int *status)
    void ctx_set_float "fsps_context_set_float" (int handle, const char *key, double value,
                                                 int *status)
    void ctx_set_str "fsps_context_set_str" (int handle, const char *key, const char *value,
                                             int *status)
    void ctx_set_fast_mode "fsps_context_set_fast_mode" (int handle, int fast_mode, int *status)

    void ctx_get_nspec "fsps_context_get_nspec" (int handle, int *n, int *status)
    void ctx_get_ntfull "fsps_context_get_ntfull" (int handle, int *n, int *status)
    void ctx_get_nbands "fsps_context_get_nbands" (int handle, int *n, int *status)
    void ctx_get_nindx "fsps_context_get_nindx" (int handle, int *n, int *status)
    void ctx_get_nz "fsps_context_get_nz" (int handle, int *n, int *status)
    void ctx_get_nemline "fsps_context_get_nemline" (int handle, int *n, int *status)
    void ctx_get_zsol "fsps_context_get_zsol" (int handle, double *zsol, int *status)
    void ctx_get_paths "fsps_context_get_paths" (int handle, char *sps_home, int sps_len,
                                                 char *data_home, int data_len,
                                                 char *output_home, int out_len)

    void ctx_get_lambda "fsps_context_get_lambda" (int handle, double *x, int *status)
    void ctx_get_ages "fsps_context_get_ages" (int handle, double *x, int *status)
    void ctx_get_zlegend "fsps_context_get_zlegend" (int handle, double *x, int *status)
    void ctx_get_emlambda "fsps_context_get_emlambda" (int handle, double *x, int *status)
    void ctx_get_filter_data "fsps_context_get_filter_data" (int handle, double *wave_eff,
                                                             double *mag_vega, double *mag_sun,
                                                             int *status)
    void ctx_set_sfh_tab "fsps_context_set_sfh_tab" (int handle, int ntab, const double *age,
                                                     const double *sfr, const double *met,
                                                     int *status)

    void ctx_compute_ssp "fsps_context_compute_ssp" (int handle, double *spec, double *mass,
                                                     double *lbol, int *status)
    void ctx_compute_csp "fsps_context_compute_csp" (int handle, int zcontinuous, int *n_out,
                                                     int *status)
    void ctx_get_csp "fsps_context_get_csp" (int handle, double *age, double *mass, double *lbol,
                                             double *sfr, double *mdust, double *mformed,
                                             double *spec, double *mags, double *emlines,
                                             double *indx, int *status)

    void ctx_compute_csp_batch "fsps_context_compute_csp_batch" (
        int handle, int n, int n_params, const char *const *names, const double *const *values,
        const double *ssp_weights, double *mags, double *spec, double *mass, double *lbol,
        double *mdust, double *sfr, double *emlines, double *indx, int *status)
    void ctx_batch_spec_device "fsps_context_batch_spec_device" (int handle, void **ptr,
                                                                 int *nspec, int *n, int *bytes,
                                                                 int *status)
    void ctx_batch_spec_host "fsps_context_batch_spec_host" (int handle, void **ptr, int *nspec,
                                                             int *n, int *bytes, int *status)
