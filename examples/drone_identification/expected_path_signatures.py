"""
This module implements code for computing the path signatures of simulated radio waves
reflected off the surface of drone and non-drone objects. For a detailed description of the
modelling approach and associated mathematical quantities, please refer to the accompanying
Jupyter notebook drone_identification.ipynb.
"""

import hashlib
import json
import multiprocessing
import os
import pickle

import esig.tosig
import joblib
import numpy as np
import scipy.constants

# This is how to define a decorator function in Python.
# See https://wiki.python.org/moin/PythonDecorators.
# We use this function to cache the results of calls to
# compute_for_drone() and compute_for_nondrone().
def cache_result(function_to_cache, cache_directory='cached_signatures'):
    """
    Cache the result of calling function_to_cache().
    """
    if not os.path.isdir(cache_directory):
        os.mkdir(cache_directory)

    def _read_result(parameter_hash):
        cache_file = os.path.join(cache_directory, str(parameter_hash))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _write_result(parameter_hash, result):
        cache_file = os.path.join(cache_directory, str(parameter_hash))
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    def wrapper_function(*args, **kwargs):
        # Aggregate all parameters passed to function_to_cache.
        # Note that args[0] will contain an instance of ExpectedSignatureCalculator(), which
        # we subsequently convert to a string. In this way, we cache results, while
        # considering values of constants and the truncation level.
        parameters = args + tuple(kwargs[k] for k in sorted(kwargs.keys()))
        # Map aggregated parameters to an MD5 hash.
        parameter_hash = hashlib.md5(json.dumps(parameters, sort_keys=True,
                                                default=str).encode('utf-8')).hexdigest()

        result = _read_result(parameter_hash)
        if result is None:
            # Call function_to_cache if no cached result is available
            result = function_to_cache(*args, **kwargs)
            _write_result(parameter_hash, result)

        return result

    return wrapper_function

class ExpectedSignatureCalculator():
    """
    Class for computing the expected path signatures of drone and non-drone objects.
    """
    # TODO Variable name -- review r0 vs r (see markdown formulae)
    # TODO Introduce additive noise?

    # Define constants
    C = scipy.constants.speed_of_light
    PI = scipy.constants.pi
    F = 10**9  # Frequency of the incident signal (1GHz)
    OMEGA = 2 * PI * F  # Angular frequency of the incident signal
    K = OMEGA / C  # Wavenumber of the incident signal
    PERIOD = 1 / F  # Period of incident signal (1e-09s)
    A = 1  # Amplitude of the incident signal
    WAVELENGTH = C / F  # Wavelength of the incident signal
    X0 = 0  # Emit and receive signals at the observer
    N_WAVELENGTHS = 1000 # Number of wavelengths per signal to generate
    N_SAMPLES = 10**5  # Number of samples per signal to generate
    T0 = 0  # Initial time

    def __init__(self, n_incident_signals, truncation_level,
                 n_processes=multiprocessing.cpu_count(),
                 use_lead_lag_transformation=False, signal_to_noise_ratio=40):
        # TODO Omit default parameter for SNR?
        """
        Parameters
        ----------
        n_incident_signals : int
            Number of incident signals to generate.
        truncation_level : int
            Path signature trucation level.
        n_processes : int
            Desired number number of parallel processes. The default is
            multiprocessing.cpu_count().
        use_lead_lag_transformation : bool
            Whether to apply the partial lead-lag transformation.
        signal_to_noise_ratio : float
            Signal-to-noise ratio with respect to the incident signal, in decibels.
        """
        self.n_incident_signals = n_incident_signals
        self.truncation_level = truncation_level
        self.n_processes = n_processes
        self.use_lead_lag_transformation = use_lead_lag_transformation

        # Time interval within which we consider incident signal
        t1 = self.X0 / self.C
        t2 = t1 + self.N_WAVELENGTHS * self.PERIOD
        # Timestamps at which we consider incident signal
        t = np.linspace(t1, t2, self.N_SAMPLES)

        self.incident_signal = self.A * np.sin(self.OMEGA * t - self.K * self.X0)

        # TODO need to mention how this works in documentation.
        incident_signal_power = self.A ** 2 / 2
        self.noise_signal_power = incident_signal_power / (10 ** (signal_to_noise_ratio / 10))

    @cache_result
    def compute_for_drone(self, rpm, speed, d, z, proportion):
        """
        Estimate the expected signature of our drone model.

        Parameters
        ----------
        rpm : float
            Number of rotations per minute of the drone's propeller.
        speed : float
            Drone's speed.
        d : float
            Diameter of the drone's propeller blades.
        z : float
            Drone's distance in relation to the observer.
        proportion : float
            Proportion of signals which hit the drone's body.
        """
        # Number of incident signals that reflect off drone body
        n_body_hits = int(self.n_incident_signals * proportion)
        # Number of incident signals that reflect off drone propeller
        n_propeller_hits = self.n_incident_signals - n_body_hits

        # Array of signatures obtained from incident signals reflected off propeller
        propeller_signatures = self._repeat_in_parallel(self._compute_propeller_signature,
                                                        n_propeller_hits,
                                                        speed, rpm, z, d)

        # Array of signatures from obtained incident signals reflected off body
        body_signatures = self._repeat_in_parallel(self._compute_body_signature,
                                                   n_body_hits,
                                                   speed, z)

        # Estimate expected signature using empirical mean
        return np.mean(np.vstack((propeller_signatures, body_signatures)), axis=0)

    @cache_result
    def compute_for_nondrone(self, speed, z):
        """
        Compute the expected signature of a non-drone object.

        Parameters
        ----------
        speed : float
            Static object's speed.
        z : float
            Static object's distance in relation to the observer.
        """
        # Array of signatures from obtained incident signals reflected off body
        body_signatures = self._repeat_in_parallel(self._compute_body_signature,
                                                   self.n_incident_signals,
                                                   speed, z)

        # Estimate expected signature using empirical mean
        return np.mean(body_signatures, axis=0)

    def _repeat_in_parallel(self, function_to_repeat, n_repetitions, *function_args):
        return joblib.Parallel(n_jobs=self.n_processes)(
            joblib.delayed(function_to_repeat)(*function_args) for i in range(n_repetitions))

    def _compute_propeller_signature(self, speed, rpm, z, d):
        # Random angle sampled uniformly from [0, 360)
        theta = np.array(np.random.uniform(0, 360))
        # Random position sampled uniformly from [0, d/2), where d/2 is blade length
        p = np.array(np.random.uniform(0, d/2))

        reflected_signal = self._compute_reflected_signal(speed, rpm, z, d, theta, p)

        return self._compute_path_signature(self.incident_signal, reflected_signal)

    def _compute_body_signature(self, speed, z):
        reflected_signal = self._compute_reflected_signal(speed=speed, rpm=0, z=z, d=1,
                                                          theta=0, p=0)

        return self._compute_path_signature(self.incident_signal, reflected_signal)

    def _compute_reflected_signal(self, speed, rpm, z, d, theta, p):
        if np.isclose(theta, 90) or np.isclose(theta, 270):
            # Case when signal hits the end of propeller blade
            v = speed
            r_0 = z - d / 2
        else:
            v = speed + p * (rpm / 60) * self.PI * 2
            r_0 = p * np.sin(np.deg2rad(theta)) + z

        # Scaling coefficient resulting from Doppler shift
        s = (1 - v / self.C) / (1 + v / self.C)

        # Time interval within which we consider reflected signal
        t3 = (-(self.X0 / self.C) +
              ((2 * (r_0 - v * self.T0)) / (s * self.C * (1 + (v / self.C)))))
        newperiod = self.PERIOD / s
        t4 = t3 + self.N_WAVELENGTHS * newperiod
        # Timestamps at which we consider reflected signal
        t = np.linspace(t3, t4, self.N_SAMPLES)

        reflected_signal = (-s * self.A) * np.sin(s * (self.OMEGA * t + self.K * self.X0) -
                                                  2 * self.K * (r_0 - v * self.T0) /
                                                  (1 + v / self.C))

        # TODO
        # Introduce additive Gaussian white noise
        # reflected_signal += np.random.randn(*reflected_signal.shape) *
        # np.sqrt(self.noise_signal_power)

        return reflected_signal

    def _compute_path_signature(self, incident_signal, reflected_signal):
        path = np.column_stack((incident_signal, reflected_signal))

        if self.use_lead_lag_transformation:
            # Compute (partial) lead-lag transformation
            path = np.repeat(path, 2, axis=0)
            path = np.column_stack((path[1:, :], path[:-1, 1]))

        return esig.tosig.stream2sig(path, self.truncation_level)

    def __str__(self):
        """
        Convert object to string. Used in conjunction with cache_result().
        """
        return str((self.C,
                    self.PI,
                    self.F,
                    self.OMEGA,
                    self.K,
                    self.PERIOD,
                    self.A,
                    self.WAVELENGTH,
                    self.X0,
                    self.N_WAVELENGTHS,
                    self.N_SAMPLES,
                    self.T0,
                    self.n_incident_signals,
                    self.truncation_level,
                    self.use_lead_lag_transformation,
                    self.noise_signal_power))
