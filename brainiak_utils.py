import math
import numpy as np

def _double_gamma_hrf(response_delay=6,
                      undershoot_delay=12,
                      response_dispersion=0.9,
                      undershoot_dispersion=0.9,
                      response_scale=1,
                      undershoot_scale=0.035,
                      temporal_resolution=100.0,
                      ):
    """Create the double gamma HRF with the timecourse evoked activity.
    Default values are based on Glover, 1999 and Walvaert, Durnez,
    Moerkerke, Verdoolaege and Rosseel, 2011
    Parameters
    ----------
    response_delay : float
        How many seconds until the peak of the HRF
    undershoot_delay : float
        How many seconds until the trough of the HRF
    response_dispersion : float
        How wide is the rising peak dispersion
    undershoot_dispersion : float
        How wide is the undershoot dispersion
    response_: float
         How big is the response relative to the peak
    undershoot_scale :float
        How big is the undershoot relative to the trough
    scale_function : bool
        Do you want to scale the function to a range of 1
    temporal_resolution : float
        How many elements per second are you modeling for the stimfunction
    Returns
    ----------
    hrf : multi dimensional array
        A double gamma HRF to be used for convolution.
    """

    hrf_length = 30  # How long is the HRF being created

    # How many seconds of the HRF will you model?
    hrf = [0] * int(hrf_length * temporal_resolution)

    # When is the peak of the two aspects of the HRF
    response_peak = response_delay * response_dispersion
    undershoot_peak = undershoot_delay * undershoot_dispersion

    for hrf_counter in list(range(len(hrf) - 1)):

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = math.pow((hrf_counter / temporal_resolution) /
                            response_peak, response_delay)
        resp_exp = math.exp(-((hrf_counter / temporal_resolution) -
                              response_peak) /
                            response_dispersion)

        response_model = response_scale * resp_pow * resp_exp

        undershoot_pow = math.pow((hrf_counter / temporal_resolution) /
                                  undershoot_peak,
                                  undershoot_delay)
        undershoot_exp = math.exp(-((hrf_counter / temporal_resolution) -
                                    undershoot_peak /
                                    undershoot_dispersion))

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For this time point find the value of the HRF
        hrf[hrf_counter] = response_model - undershoot_model

    return hrf

def convolve_hrf(stimfunction,
                 tr_duration,
                 hrf_type='double_gamma',
                 scale_function=True,
                 temporal_resolution=100.0,
                 ):
    """ Convolve the specified hrf with the timecourse.
    The output of this is a downsampled convolution of the stimfunction and
    the HRF function. If temporal_resolution is 1 / tr_duration then the
    output will be the same length as stimfunction. This time course assumes
    that slice time correction has occurred and all slices have been aligned
    to the middle time point in the TR.
    Be aware that if scaling is on and event durations are less than the
    duration of a TR then the hrf may or may not come out as anticipated.
    This is because very short events would evoke a small absolute response
    after convolution  but if there are only short events and you scale then
    this will look similar to a convolution with longer events. In general
    scaling is useful, which is why it is the default, but be aware of this
    edge case and if it is a concern, set the scale_function to false.
    Parameters
    ----------
    stimfunction : timepoint by feature array
        What is the time course of events to be modelled in this
        experiment. This can specify one or more timecourses of events.
        The events can be weighted or binary
    tr_duration : float
        How long (in s) between each volume onset
    hrf_type : str or list
        Takes in a string describing the hrf that ought to be created.
        Can instead take in a vector describing the HRF as it was
        specified by any function. The default is 'double_gamma' in which
        an initial rise and an undershoot are modelled.
    scale_function : bool
        Do you want to scale the function to a range of 1
    temporal_resolution : float
        How many elements per second are you modeling for the stimfunction
    Returns
    ----------
    signal_function : timepoint by timecourse array
        The time course of the HRF convolved with the stimulus function.
        This can have multiple time courses specified as different
        columns in this array.
    """

    # Check if it is timepoint by feature
    if stimfunction.shape[0] < stimfunction.shape[1]:
        logger.warning('Stimfunction may be the wrong shape')

    # How will stimfunction be resized
    stride = int(temporal_resolution * tr_duration)
    duration = int(stimfunction.shape[0] / stride)

    # Generate the hrf to use in the convolution
    if hrf_type == 'double_gamma':
        hrf = _double_gamma_hrf(temporal_resolution=temporal_resolution)
    elif isinstance(hrf_type, list):
        hrf = hrf_type

    # How many timecourses are there
    list_num = stimfunction.shape[1]

    # Create signal functions for each list in the stimfunction
    for list_counter in range(list_num):

        # Perform the convolution
        signal_temp = np.convolve(stimfunction[:, list_counter], hrf)

        # Down sample the signal function so that it only has one element per
        # TR. This assumes that all slices are collected at the same time,
        # which is often the result of slice time correction. In other
        # words, the output assumes slice time correction
        signal_temp = signal_temp[:duration * stride]
        signal_vox = signal_temp[int(stride / 2)::stride]

        # Scale the function so that the peak response is 1
        if scale_function:
            signal_vox = signal_vox / np.max(signal_vox)

        # Add this function to the stack
        if list_counter == 0:
            signal_function = np.zeros((len(signal_vox), list_num))

        signal_function[:, list_counter] = signal_vox

    return signal_function