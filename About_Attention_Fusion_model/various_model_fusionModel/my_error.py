class DataFormatError(Exception):
    def __init__(self, data_format):
        err = 'The data_format "{}" is illegal'.format(data_format)
        Exception.__init__(self, err)
        self.data_format = data_format


class ModelNameError(Exception):
    def __init__(self, model_name):
        err = 'there is no model named {} , please check the model_name'.format(model_name)
        Exception.__init__(self, err)
        self.model_name = model_name

