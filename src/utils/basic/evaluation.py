

class PairedDomainLogAnalyzer(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.knn_acc_dicts = []
        self.reconstruction_error_dict = {}
        self.latent_distances = []
        self.experiment_type = None

        self.analyze_log_file()

    def analyze_log_file(self):
        with open(self.log_file) as f:
            lines = f.readlines()
            test_metrics = False
            for line in lines:
                line = line.lower()
                # Filter for test statistics
                if 'test loss statistics' in line:
                    test_metrics = True
                    self.knn_acc_dicts.append({})
                elif 'train' in line or 'val' in line:
                    test_metrics = False

                if test_metrics:
                    if 'reconstruction loss' in line:
                        words = line.split()
                        domain = words[words.index('domain:')-1]
                        score = float(words[-1])
                        if domain not in self.reconstruction_error_dict:
                            self.reconstruction_error_dict[domain] = []
                        self.reconstruction_error_dict[domain].append(score)
                    if 'latent l1 distance' in line:
                        score = float(line.split()[-1])
                        self.latent_distances.append(score)
                    if '-nn accuracy' in line.lower():
                        idx = line.index('-nn accuracy')
                        k = int(line[idx-2:idx])
                        score = float(line.split()[-1])
                        self.knn_acc_dicts[-1][k] = score
        if len(self.latent_distances) > 1:
            self.experiment_type = 'cv'
        else:
            self.experiment_type='train_val_test'

