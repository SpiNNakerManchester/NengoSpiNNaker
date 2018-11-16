

class NengoLiveOutputInterface(object):

    def output(self, t, x):
        """ enforced by the nengo_spinnaker_gfe duck typing
    
        :param t:
        :param x:
        :return:
        """
        pass

    @property
    def size_in(self):
        """
        
        :return: 
        """
        return None

    @property
    def size_out(self):
        """
        
        :return: 
        """
        return None


