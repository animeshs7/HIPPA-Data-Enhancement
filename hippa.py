!pip install pycryptodome
!pip install diffprivlib
import random;
from Crypto.Hash import keccak
import pandas as pd;
import hashlib
healthcare=pd.read_csv("Hospital ER.csv");
from numbers import Real
import numpy as np
from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring
class Laplace(DPMechanism):
    def __init__(self, *, epsilon, delta=0.0, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None
    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        return float(sensitivity)
    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)
        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")
        return True
    def bias(self, value):
        return 0.0
    def variance(self, value):
        self._check_all(0)
        return 2 * (self.sensitivity / (self.epsilon - np.log(1 -self.delta))) ** 2
    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 -unif3) * np.cos(np.pi * unif4)
    def randomise(self, value):
        self._check_all(value)
        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(self._rng.random(),self._rng.random(), self._rng.random(),self._rng.random())
        return value - scale * standard_laplace
class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta,sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)
    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)
        shape = self.sensitivity / self.epsilon
        return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))
    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value) 
        shape = self.sensitivity / self.epsilon
        variance = value ** 2 + shape * (self.lower * np.exp((self.lower- value) / shape)- self.upper * np.exp((value -self.upper) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower - value) /shape)- np.exp((value - self.upper) /shape))
        variance -= (self.bias(value) + value) ** 2
        return variance
    def _check_all(self, value):
        Laplace._check_all(self, value)
        TruncationAndFoldingMixin._check_all(self, value)
        return True
    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)
        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)
class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta,sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower,upper=upper)
    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)
        shape = self.sensitivity / self.epsilon 
        bias = shape * (np.exp((self.lower + self.upper - 2 * value) /shape) - 1)
        bias /= np.exp((self.lower - value) / shape) + np.exp((self.upper- value) / shape)
        return bias
    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError
    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)
        return True
    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)
        noisy_value = super().randomise(value)
        return self._fold(noisy_value)
class LaplaceBoundedDomain(LaplaceTruncated):
    def _find_scale(self):
        eps = self.epsilon
        delta = self.delta
        diam = self.upper - self.lower
        delta_q = self.sensitivity
        def _delta_c(shape):
            if shape == 0:
                return 2.0
            return (2 - np.exp(- delta_q / shape) - np.exp(- (diam -delta_q) / shape)) / (1 - np.exp(- diam / shape))
        def _f(shape):
            return delta_q / (eps - np.log(_delta_c(shape)) - np.log(1 - delta))
            left = delta_q / (eps - np.log(1 - delta))
            right = _f(left)
            old_interval_size = (right - left) * 2
            while old_interval_size > right - left:
                old_interval_size = right - left
                middle = (right + left) / 2
                if _f(middle) >= middle:
                    left = middle
                if _f(middle) <= middle:
                    right = middle
            return (right + left) / 2
    def effective_epsilon(self):
        if self._scale is None:
            self._scale = self._find_scale()
        if self.delta > 0.0:
            return None
        return self.sensitivity / self._scale
    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)
        if self._scale is None:
            self._scale = self._find_scale()
        bias = (self._scale - self.lower + value) / 2 * np.exp((self.lower - value) / self._scale) - (self._scale + self.upper - value) / 2 * np.exp((value -self.upper) / self._scale)
        bias /= 1 - np.exp((self.lower - value) / self._scale) / 2 - np.exp((value - self.upper) / self._scale) / 2
        return bias
    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)
        if self._scale is None:
            self._scale = self._find_scale()
        variance = value**2
        variance -= (np.exp((self.lower - value) / self._scale) * (self.lower ** 2) + np.exp((value - self.upper) / self._scale) * (self.upper ** 2)) / 2
        variance += self._scale * (self.lower * np.exp((self.lower - value) / self._scale) - self.upper * np.exp((value -self.upper) / self._scale))
        variance += (self._scale ** 2) * (2 - np.exp((self.lower - value)/ self._scale)- np.exp((value - self.upper) /self._scale))
        variance /= 1 - (np.exp(-(value - self.lower) / self._scale) + np.exp(-(self.upper - value) / self._scale)) /2
        variance -= (self.bias(value) + value) ** 2
        return variance
    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)
        if self._scale is None:
            self._scale = self._find_scale()
        value = max(min(value, self.upper), self.lower)
        if np.isnan(value):
            return float("nan")
        samples = 1
        while True:
            try:
                unif = self._rng.random(4 * samples)
            except TypeError: # rng is secrets.SystemRandom
                unif = [self._rng.random() for _ in range(4 * samples)]
        noisy = value + self._scale *self._laplace_sampler(*np.array(unif).reshape(4, -1))
        if ((noisy >= self.lower) & (noisy <= self.upper)).any():
            idx = np.argmax((noisy >= self.lower) & (noisy <=self.upper))
            return noisy[idx]
        samples = min(100000, samples * 2)
class LaplaceBoundedNoise(Laplace):
    def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        self._noise_bound = None
    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")
        if isinstance(delta, Real) and not 0 < delta < 0.5:
            raise ValueError("Delta must be strictly in the interval(0,0.5). For zero delta, use :class:`.Laplace`.")
        return super()._check_epsilon_delta(epsilon, delta)
    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0
    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError
    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)
        if self._scale is None or self._noise_bound is None:
            self._scale = self.sensitivity / self.epsilon
            self._noise_bound = 0 if self._scale == 0 else \
            self._scale * np.log(1 + (np.exp(self.epsilon) - 1) / 2 /self.delta)
        if np.isnan(value):
            return float("nan")
        samples = 1
        while True:
            try:
                unif = self._rng.random(4 * samples)
            except TypeError: # rng is secrets.SystemRandom
                unif = [self._rng.random() for _ in range(4 * samples)]
            noisy = self._scale *self._laplace_sampler(*np.array(unif).reshape(4, -1))
            if ((noisy >= - self._noise_bound) & (noisy <= self._noise_bound)).any():
                idx = np.argmax((noisy >= - self._noise_bound) & (noisy<= self._noise_bound))
                return value + noisy[idx]
            samples = min(100000, samples * 2)
last_names=healthcare["patient_last_name"];
ids=healthcare["patient_id"];
sat=healthcare["patient_sat_score"];
sensitivity=3
epsilon=0.3
mechanism = LaplaceTruncated(epsilon=epsilon,delta=0.0,sensitivity=sensitivity,lower=0,upper=10)
new_sat=[];
for x in sat:
    new_sat.append(mechanism.randomise(x));
healthcare["patient_sat_score"]=new_sat;
age=healthcare["patient_age"];
new_age=[]
for x in age:
    if x<=15:
        new_age.append("<15");
    elif x<=30:
        new_age.append("15-30");
    elif x<=50:
        new_age.append("30-50");
    else:
        new_age.append(">50");
    healthcare["patient_age"]=new_age
def hash_unicode(a_string):
    sha_out= hashlib.sha256(a_string.encode('utf-8')).hexdigest()
    keccak_hash3 = keccak.new(digest_bits=224)
    keccak_hash3.update(sha_out.encode("utf-8"))
    hex=keccak_hash3.hexdigest();
    out="";
    for x in range(11):
        num=random.randint(0,55);
        out += hex[num];
    return out;
def supperession(a_string):
    return "XXXXXXXXX"
healthcare['patient_id']=healthcare['patient_id'].apply(hash_unicode);
healthcare['patient_last_name']=healthcare['patient_last_name'].apply(supperession)
healthcare.head()
from pathlib import Path
filepath = Path('content/subfolder/out.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
healthcare.to_csv(filepath)