
class BaseTester:
    def __init__(self):
        pass

    def get_input_data(self, results: dict):
        return results

    def get_label_data(self, results: dict):
        return results

    def get_vis(self, results):
        return results
    
    def process(self, results):
        pass


class InsSegTester(BaseTester):
    def __init__(self):
        super(InsSegTester, self).__init__()
    