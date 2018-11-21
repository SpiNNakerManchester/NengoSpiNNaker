# general imports
import time
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
        "_stopped",
        #
        "_n_machine_time_steps"
    ]

    SLEEP_PERIOD = 0.0001

    def __init__(self, host_network, time_step, n_machine_time_steps):
        self._host_network = host_network
        self._time_step = time_step
        self._running = False
        self._stopped = False
        self._in_pause_state = True
        self._n_machine_time_steps = n_machine_time_steps

        thread = Thread(name="nengo graph updater thread", target=self.run)
        thread.daemon = True
        thread.start()

    def pause_stop(self):
        self._in_pause_state = True

    def start_resume(self):
        self._in_pause_state = False

    def run(self):
        # set the fag to running
        self._running = True
        step = 0

        # check that the code is still running
        while self._running and not self._stopped:

            # check that the code is not in a pause state
            while (not self._in_pause_state and not self._stopped and step <
                    self._n_machine_time_steps):

                # time holder
                start_time = time.time()

                # Run a step
                self._host_network.step()
                step += 1

                # hang till time step over
                run_time = time.time() - start_time

                # If that step took less than timestep then hang
                time.sleep(self.SLEEP_PERIOD)
                #while run_time < self._time_step:
                #    time.sleep(self.SLEEP_PERIOD)
                #    run_time = time.time() - start_time

        self._stopped = True

    def close(self):
        self._running = False
        while not self._stopped:
            time.sleep(self.SLEEP_PERIOD)
