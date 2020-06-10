# vim: expandtab:ts=4:sw=4
from utils import VehicleClass
from parameter import nn_budget

car_class = [VehicleClass.car.value, VehicleClass.car_h.value, VehicleClass.car_nh.value, VehicleClass.car_hc.value,
             VehicleClass.taxi.value]


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Leaved = 4


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, v_class, v_score, plate, p_score,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.center_path = []

        self._n_init = n_init
        self._max_age = max_age

        self.v_class = v_class
        self.v_score = v_score

        self.plate = plate
        self.p_score = p_score

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_tlbr_int(self):

        return list(map(lambda x: int(x), self.to_tlbr()))

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        bbox = self.to_tlbr()
        self.center_path.append((int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2)))
        if len(self.center_path) > nn_budget:
            self.center_path = self.center_path[-nn_budget:]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative:
            # sepcial setting for the motor
            if self.hits >= self._n_init or (self.v_class == VehicleClass.motor.value and self.hits >= self._n_init / 4):
                self.state = TrackState.Confirmed

        if self.v_class in car_class and detection.v_class in car_class:
            if detection.v_class >= self.v_class:
                self.v_class = detection.v_class
                self.v_score = detection.v_score
        elif detection.v_score >= self.v_score:
            self.v_score = detection.v_score
            self.v_class = detection.v_class

        if detection.p_score >= self.p_score:
            self.plate = detection.plate
            self.p_score = detection.p_score

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Leaved

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted or self.state == TrackState.Leaved

    def is_leaved(self):
        """Returns True if this track is dead and should be leaved."""
        return self.state == TrackState.Leaved
