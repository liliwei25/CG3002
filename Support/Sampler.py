import numpy

class Sampler:

    def __init__(self, num_of_signals, frame_length, sampling_interval):
        self.leftover_bytestring_set = None
        self.cur_bytestring_set = None
        self.frame_length = frame_length 
        self.sampling_interval = sampling_interval
        self.num_of_signals = num_of_signals

    def sample(self, bytestring_set):
        """input bytestring set is a numpy array. 
        Returns a collection of frames, represented by numpy arrays
        """
        frame_collection = numpy.zeros((0, self.num_of_signals, self.frame_length))
        if(self.leftover_bytestring_set is not None):
            self.cur_bytestring_set = numpy.append(self.leftover_bytestring_set, bytestring_set, axis = 1)
        else:
            self.cur_bytestring_set = bytestring_set

        start = 0
        end = start + self.frame_length
        while end <= self.cur_bytestring_set.shape[1]:
            frame = numpy.copy(self.cur_bytestring_set[0:self.num_of_signals, start:end])
            frame_collection = numpy.append(frame_collection, [frame], axis = 0)
            start += self.sampling_interval
            end += self.sampling_interval

        self.leftover_bytestring_set = self.cur_bytestring_set[0:self.num_of_signals, start:self.cur_bytestring_set.shape[1]]
        return frame_collection


