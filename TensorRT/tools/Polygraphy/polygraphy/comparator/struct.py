#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections import OrderedDict

from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc


class IterationResult(OrderedDict):
    def __init__(self, outputs=None, runtime=None, runner_name=None):
        """
        An ordered dictionary containing the result of a running a single iteration of a runner.

        This maps output names to NumPy arrays, and preserves the output ordering from the runner.

        Also includes additional fields indicating the name of the runner which produced the
        outputs, and the time required to do so.


        Args:
            outputs (Dict[str, np.array]): The outputs of this iteration, mapped to their names.


            runtime (float): The time required for this iteration, in seconds.
            runner_name (str): The name of the runner that produced this output.
        """
        # IMPORTANT: This class must be pickleable.
        initial = misc.default_value(outputs, {})
        # Before 3.6, OrderedDict.update() did not preserve ordering
        for key, val in initial.items():
            self[key] = val
        self.runtime = runtime
        self.runner_name = misc.default_value(runner_name, "")


class RunResults(list):
    def __init__(self):
        """
        Maps runner names to zero or more IterationResults.

        Note: Technically, it is an List[Tuple[str, List[IterationResult]]], but includes
        helpers that make it behave like an OrderedDict that can contain duplicates.
        """
        pass


    def items(self):
        """
        Creates a generator that yields Tuple[str, List[IterationResult]] - runner names
        and corresponding outputs.
        """
        for name, iteration_results in self:
            yield name, iteration_results


    def keys(self):
        """
        Creates a generator that yields runner names (str).
        """
        for name, _ in self:
            yield name


    def values(self):
        """
        Creates a generator that yields runner outputs (List[IterationResult]).
        """
        for _, iteration_results in self:
            yield iteration_results


    def update(self, other):
        """
        Updates the results stored in this instance.

        Args:
            other (Union[Dict[str, List[IterationResult]], RunResults]):
                    A dictionary or RunResults instance from which to update this one.
        """
        for name, iteration_results in other.items():
            self[name] = iteration_results
        return self


    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)

        for name, iteration_results in self:
            if name == key:
                return iteration_results

        G_LOGGER.critical("Runner: {:} does not exist in this RunResults instance. Note: Available runners: {:}".format(
                        key, list(self.keys())))


    def __setitem__(self, key, value):
        if isinstance(key, int):
            super().__setitem__(key, value)

        for index, name in enumerate(self.keys()):
            if name == key:
                super().__setitem__(index, (key, value))
                break
        else:
            self.append((key, value))


    def __contains__(self, val):
        if isinstance(val, str) or isinstance(val, bytes):
            return val in list(self.keys())
        return super().__contains__(val)


class AccuracyResult(OrderedDict):
    """
    An ordered dictionary including details about the result of `Comparator.compare_accuracy`.

    More specifically, it is an OrderedDict[Tuple[str, str], List[OrderedDict[str, bool]]] which maps a runner
    pair (a tuple containing both runner names) to a list of dictionaries of booleans (or anything that can be
    converted into a boolean, such as an OutputCompareResult), indicating whether there was a match in the outputs of
    the corresponding iteration. The List[OrderedDict[str, bool]] is constructed from the dictionaries returned
    by `compare_func` in `compare_accuracy`.

    For example, to see if there was a match in "output0" between "runner0" and
    "runner1" on the 1st iteration, you would do the following:
    ::

        output_name = "output0"
        runner_pair = ("runner0", "runner1")
        iteration = 0
        match = bool(accuracy_result[runner_pair][iteration][output_name])

    In case there's a mismatch, you would be able to inspect the outputs in question by accessing
    the results from Comparator.run() (assumed here to be called run_results):
    ::

        runner0_output = run_results["runner0"][iteration][output_name]
        runner1_output = run_results["runner1"][iteration][output_name]
    """
    def __bool__(self):
        """
        Whether all outputs matched for every iteration.
        You can use this function to avoid manually checking each output. For example:
        ::

            if accuracy_result:
                print("All matched!")

        Returns:
            bool
        """
        return all([bool(match) for outs in self.values() for out in outs for match in out.values()])


    def _get_runner_pair(self, runner_pair):
        return misc.default_value(runner_pair, list(self.keys())[0])


    def percentage(self, runner_pair=None):
        """
        Returns the percentage of iterations that matched for the given pair of runners,
        expressed as a decimal between 0.0 and 1.0.

        Always returns 1.0 when the number of iterations is 0, or when there are no runner comparisons.

        Args:
            runner_pair ((str, str)):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.
        """
        if not list(self.keys()):
            return 1.0 # No data in this result.

        matched, _, total = self.stats(runner_pair)
        if not total:
            return 1.0 # No iterations
        return float(matched) / float(total)


    def stats(self, runner_pair=None):
        """
        Returns the number of iterations that matched, mismatched, and the total number of iterations.

        Args:
            runner_pair ((str, str)):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            (int, int, int): Number of iterations that matched, mismatched, and total respectively.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        outs = self[runner_pair]
        matched = sum([all([match for match in out.values()]) for out in outs])
        total = len(outs)
        return matched, total - matched, total
