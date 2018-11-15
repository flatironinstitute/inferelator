import numpy as np
import pandas as pd

from inferelator_ng import utils

class BaseRegression:
    # These are all the things that have to be set in a new regression class

    # Variables that handle multiprocessing via SLURM / KVS
    # The defaults here are placeholders for troubleshooting
    # These should always be provided when instantiating
    kvs = None  # KVSClient
    chunk = None  # int

    # Raw Data
    X = None  # [K x N] float
    Y = None  # [G x N] float
    G = None  # int G
    K = None  # int K

    def run(self):
        raise NotImplementedError

    @staticmethod
    def _scale(df):
        """
        Center and normalize a DataFrame
        :param df: pd.DataFrame
        :return df: pd.DataFrame
        """
        df = df.T
        return ((df - df.mean()) / df.std(ddof=1)).T

    def pileup_data(self):
        """
        Take the completed run data and pack it up into a DataFrame of betas
        :return: (pd.DataFrame [G x K], pd.DataFrame [G x K])
        """
        run_data = []

        # Reach into KVS to get the model data
        for p in range(utils.slurm_envs()['tasks']):
            pid, ps = self.kvs.get('plist')
            run_data.extend(ps)
            print("Collected {l} models from proc {id}".format(l=len(ps), id=pid))
        self.kvs.finish_own_check()

        # Create G x K arrays of 0s to populate with the regression data
        betas = np.zeros((self.G, self.K), dtype=np.dtype(float))
        betas_rescale = np.zeros((self.G, self.K), dtype=np.dtype(float))

        # Populate the zero arrays with the BBSR betas
        for data in run_data:
            xidx = data['ind']  # Int
            yidx = data['pp']  # Boolean array of size K

            betas[xidx, yidx] = data['betas']
            betas_rescale[xidx, yidx] = data['betas_resc']

        d_len, b_avg, null_m = self._summary_stats(betas)
        utils.Debug.vprint("Regression complete:", end=" ", level=0)
        utils.Debug.vprint("{d_len} Models, {b_avg} Preds per Model, {nom} Null Models".format(d_len=d_len,
                                                                                               b_avg=round(b_avg, 4),
                                                                                               nom=null_m), level=0)

        # Convert arrays into pd.DataFrames to return results
        betas = pd.DataFrame(betas, index=self.Y.index, columns=self.X.index)
        betas_rescale = pd.DataFrame(betas_rescale, index=self.Y.index, columns=self.X.index)

        return betas, betas_rescale

    @staticmethod
    def _summary_stats(arr):
        d_len = arr.shape[0]
        b_avg = np.mean(np.sum(arr != 0, axis=1))
        null_m = np.sum(np.sum(arr != 0, axis=1) == 0)
        return d_len, b_avg, null_m