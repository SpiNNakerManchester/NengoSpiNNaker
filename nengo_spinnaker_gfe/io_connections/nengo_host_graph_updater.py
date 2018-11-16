# general imports
import time
from threading import Condition
from threading import Thread


class NengoHostGraphUpdater(object):

    __slots__ = [
        #
        '_host_network',
        #
        '_time_step',
        #
        '_running',
        #
        '_running_condition',
        #
        "_in_pause_state",
        #
        "_in_pause_condition",
        #
        "_stopped_condition",
        #
        "_stopped"
    ]

    SLEEP_PERIOD = 0.0001

    def __init__(self, host_network, time_step):
        self._host_network = host_network
        self._time_step = time_step
        self._running = False
        self._stopped = False
        self._in_pause_state = True
        self._running_condition = Condition()
        self._in_pause_condition = Condition()
        self._stopped_condition = Condition()

        thread = Thread(name="nengo graph updater thread", target=self.run)
        thread.daemon = True
        thread.start()

    def pause_stop(self):
        self._in_pause_condition.acquire()
        self._in_pause_state = True
        self._in_pause_condition.release()

    def start_resume(self):
        self._in_pause_condition.acquire()
        self._in_pause_state = False
        self._in_pause_condition.release()

    def run(self):
        # set the fag to running
        self._running_condition.acquire()
        self._running = True
        self._running_condition.release()

        # check that the code is still running
        self._running_condition.acquire()
        while self._running:
            self._running_condition.release()

            # check that the code is not in a pause state
            self._in_pause_condition.acquire()
            while not self._in_pause_state:
                self._in_pause_condition.release()

                # time holder
                start_time = time.time()

                # Run a step
                self._host_network.step()

                # hang till time step over
                run_time = time.time() - start_time

                # If that step took less than timestep then hang
                time.sleep(self.SLEEP_PERIOD)
                while run_time < self._time_step:
                    time.sleep(self.SLEEP_PERIOD)
                    run_time = time.time() - start_time
                self._in_pause_condition.acquire()

            # release pause condition to ensure its complete
            self._in_pause_condition.release()
            # acquire running lock to ensure its locked when while check is done
            self._running_condition.acquire()

        self._stopped_condition.acquire()
        self._stopped = True
        self._stopped_condition.release()

    def close(self):
        self._running_condition.acquire()
        self._running = False
        self._running_condition.release()

        self._stopped_condition.acquire()
        while not self._stopped:
            self._stopped_condition.release()
            time.sleep(self.SLEEP_PERIOD)
            self._stopped_condition.acquire()
        self._stopped_condition.release()
