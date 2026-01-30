import os.path
import pandas as pd
from spglm.glm import GLM
from spglm.family import Gaussian
import matplotlib.pyplot as plt
import csv
import datetime
import math
import numpy as np
from scipy.spatial.distance import cdist


class ALPHA_OPT(GLM):
    def __init__(self, coords, y, X, bw, data, variables, comm, n_n, bt_value, fname, alphatype=False, adaptiveBG=False,
                 family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel=False, fixed=False, constant=True,
                 spherical=False, hat_matrix=False):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        self.alphatype = alphatype
        self.adaptiveBG = adaptiveBG
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}
        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.spherical = spherical
        self.hat_matrix = hat_matrix
        self.data = data
        self.variables = variables
        self.bt_value = bt_value
        self.iter = n_n
        self.comm = comm
        self.fname = fname

        self.k = self.X.shape[1]

        m_m = int(math.ceil(float(len(self.iter)) / self.comm.size))
        self.x_chunk = self.iter[self.comm.rank * m_m:(self.comm.rank + 1) * m_m]

    def weight_func(self, i, bw):
        """ This section calculate the weight matrix based on data similarity. We have (i) obeservation and we need to estimate
        its similarity with the entire dataset """

        """ Calculate the weight matrix based on data similarity. """
        if self.alphatype:
            dist = cdist([self.coords[i]], self.coords).reshape(-1)

            # fixed gaussian
            if self.fixed:
                # wi = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
                if self.adaptiveBG:
                    eps = 1e-10  # small value to prevent division by zero
                    bw = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                    wi = np.exp(-0.5 * (dist / (bw + eps)) ** 2).reshape(-1, 1)
                else:
                    wi = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
            # adaptive bisquare
            else:
                maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                zs = dist / maxd
                zs[zs >= 1] = 1
                wi = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)
        else:
            ### Similarity weight matrix
            data_matrix = self.data[self.variables].values
            data_point = data_matrix[i, :]
            diff = np.abs(data_matrix - data_point)
            combined = np.mean(diff, axis=1)
            w2 = np.exp(-combined ** 2)
            ws = w2.reshape(-1, 1)

            ### Geographically weighted
            dist = cdist([self.coords[i]], self.coords).reshape(-1)

            # fixed gaussian
            if self.fixed:
                # wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
                if self.adaptiveBG:
                    eps = 1e-10  # small value to prevent division by zero
                    bw = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                    wg = np.exp(-0.5 * (dist / (bw + eps)) ** 2).reshape(-1, 1)
                else:
                    wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
            # adaptive bisquare
            else:
                maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                zs = dist / maxd
                zs[zs >= 1] = 1
                wg = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)

            wi = self.bt_value * wg + (1 - self.bt_value) * ws

        return wi

    #### based on fastgwr
    def local_fitting(self, i):
        wi = self.weight_func(i, self.bw).reshape(-1, 1)
        xT = (self.X * wi).T
        try:
            xtx_inv_xt = np.dot(np.linalg.inv(np.dot(xT, self.X)), xT)
        except np.linalg.LinAlgError as e:
            xtx_inv_xt = np.dot(np.linalg.pinv(np.dot(xT, self.X)), xT)
        betas = np.dot(xtx_inv_xt, self.y).reshape(-1)

        ri = np.dot(self.X[i], xtx_inv_xt)
        predy = np.dot(self.X[i], betas)
        err = self.y[i][0] - predy
        CCT = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T))

        return np.concatenate(([i, err, ri[i]], betas, CCT))

    def alpha_opt_mpi(self):
        sub_Betas = np.empty((self.x_chunk.shape[0], 2 * self.k + 3), dtype=np.float64)
        pos = 0
        for i in self.x_chunk:
            sub_Betas[pos] = self.local_fitting(i)
            pos += 1

        Betas_list = self.comm.gather(sub_Betas, root=0)
        if self.comm.rank == 0:
            data = np.vstack(Betas_list)

            RSS = np.sum(data[:, 1] ** 2)
            TSS = np.sum((self.y - np.mean(self.y)) ** 2)
            R2 = 1 - RSS / TSS
            trS = np.sum(data[:, 2])

            sigma2_v1 = RSS / (self.n - trS)
            aicc = self.n * np.log((RSS) / (self.n)) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                    self.n - trS - 2.0)
            data[:, -self.k:] = np.sqrt(data[:, -self.k:] * sigma2_v1)

            Adj_R2 = 1 - (1 - R2) * (self.n - 1) / (self.n - trS - 1)

            print(f"R2: {R2:.3f}", flush=True)
            print(f"Adj-R2: {Adj_R2:.3f}", flush=True)
            print(f"AICc: {aicc:.3f}", flush=True)

            directory, filename = os.path.split(self.fname)
            base_filename, _ = os.path.splitext(filename)
            new_filename = f"{base_filename}_Results.csv"
            new_path = os.path.join(directory, new_filename)

            varNames = np.genfromtxt(self.fname, dtype=str, delimiter=',', names=True, max_rows=1).dtype.names[3:]
            header = "index, residual,influ,"
            if self.constant:
                varNames = ['intercept'] + list(varNames)
            for x in varNames:
                header += ("b_" + x + ',')
            for x in varNames:
                header += ("se_" + x + ',')
            np.savetxt(new_path, data, delimiter=',', header=header[:-1], comments='')


class SGWR:
    def __init__(self, comm, parser):
        """
        Class Initialization
        """
        self.start_time = datetime.datetime.now()
        self.comm = comm
        self.parser = parser
        self.X = None
        self.y = None
        self.coords = None
        self.n = None
        self.k = None
        self.iter = None
        self.minbw = None
        self.maxbw = None
        self.bw = None
        self.kernel = None
        self.adaptiveBG = None
        self.parse_sgwr_args()

        if self.comm.rank == 0:
            self.read_file()
            self.k = self.X.shape[1]
            self.iter = np.arange(self.n)

        # Broadcast data to all processors
        self.X = comm.bcast(self.X, root=0)
        self.y = comm.bcast(self.y, root=0)
        self.coords = comm.bcast(self.coords, root=0)
        self.iter = comm.bcast(self.iter, root=0)
        self.n = comm.bcast(self.n, root=0)
        self.k = comm.bcast(self.k, root=0)

        # Divide iterations among processes
        m = int(math.ceil(float(len(self.iter)) / self.comm.size))
        self.x_chunk = self.iter[self.comm.rank * m: (self.comm.rank + 1) * m]

    def parse_sgwr_args(self):
        """
        Parse command-line arguments.
        """
        parser_arg = self.parser.parse_args()
        self.fname = parser_arg.data
        self.fout = parser_arg.out
        self.fixed = parser_arg.fixed
        self.constant = parser_arg.constant
        self.estonly = parser_arg.estonly
        self.standardize = parser_arg.standardize
        self.gwr_model = parser_arg.gwr
        self.alphacurve = parser_arg.alphacurve

        # Kernel logic
        if parser_arg.fixed:
            self.kernel = 'fixed'
            self.fixed = True
        elif parser_arg.biga:
            self.kernel = 'adaptive'
            self.fixed = False
        else:
            self.kernel = 'adaptive'
            self.fixed = False

        if parser_arg.bw:
            self.bw = float(parser_arg.bw) if self.fixed else int(parser_arg.bw)
        if parser_arg.minbw:
            self.minbw = float(parser_arg.minbw) if self.fixed else int(parser_arg.minbw)

        if self.comm.rank == 0:
            print("-" * 60, flush=True)
            if self.fixed:
                print("Spatial Kernel: Fixed Gaussian", flush=True)
            else:
                print("Spatial Kernel: Adaptive Bisquare", flush=True)

    def read_file(self):
        """
        Read file from the provided path.
        """
        input_data = np.genfromtxt(self.fname, dtype=float, delimiter=',', skip_header=True)
        self.y = input_data[:, 2].reshape(-1, 1)
        self.n = input_data.shape[0]
        self.X = input_data[:, 3:]  # Use columns starting at index 3 as predictors.

        if self.standardize:
            self.y = (self.y - np.mean(self.y, axis=0)) / np.std(self.y, axis=0)
            self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

        self.X = np.hstack([np.ones((self.n, 1)), self.X])  ## adding the intercept

        self.coords = input_data[:, :2]

    def set_search_range(self):
        """
        Define the search range for the bandwidth in the golden section search.
        """
        if self.fixed:
            max_dist = np.max(np.array([np.max(cdist([self.coords[i]], self.coords))
                                        for i in range(self.n)]))
            self.maxbw = max_dist * 2
            if self.minbw is None:
                min_dist = np.min(
                    np.array([np.min(np.delete(cdist(self.coords[[i]], self.coords), i))
                              for i in range(self.n)]))
                self.minbw = min_dist / 2
                if self.minbw < 0.01:  ##added Naser
                    self.minbw = 0.01
        else:
            self.maxbw = self.n
            if self.minbw is None:
                self.minbw = 40 + 2 * self.k

        print(f'Min bw: {self.minbw:.3f}, Max bw: {self.maxbw:.3f}', flush=True)

    def _local_fit(self, i, y, X, bw):
        # Compute distances for observation i
        dist = cdist([self.coords[i]], self.coords).reshape(-1)
        if self.fixed:
            wi = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
        else:
            maxdis = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
            wg = dist / maxdis
            wg[wg >= 1] = 1
            wi = ((1 - (wg) ** 2) ** 2).reshape(-1, 1)

        # Form the weighted design matrix and response vector
        X_new = X * np.sqrt(wi)
        Y_new = y * np.sqrt(wi)

        # Regularization: add a tiny constant to the diagonal before inverting
        epsilon = 1e-8
        A = np.dot(X_new.T, X_new) + epsilon * np.eye(X_new.shape[1])
        invA = np.linalg.inv(A)
        temp = np.dot(invA, X_new.T)
        hat = np.dot(X_new[i], temp[:, i])
        yhat = np.sum(np.dot(X_new, temp[:, i]).reshape(-1, 1) * Y_new)
        err = Y_new[i][0] - yhat

        return err ** 2, hat

    def local_fit(self, i, y, X, bw):
        """
        MPI and serial versions now use the same _local_fit method.
        """
        return self._local_fit(i, y, X, bw)

    def golden_section(self, a, c, function):
        delta = 0.38197
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
        opt_bw = None
        score = None
        diff = 1.0e9
        iters = 0
        dict = {}
        while np.abs(diff) > 1.0e-6 and iters < 200:
            iters += 1
            if not self.fixed:
                b = np.round(b)
                d = np.round(d)

            if b in dict:
                score_b = dict[b]
            else:
                score_b = function(b)
                dict[b] = score_b

            if d in dict:
                score_d = dict[d]
            else:
                score_d = function(d)
                dict[d] = score_d

            if score_b <= score_d:
                opt_score = score_b
                opt_bw = b
                c = d
                d = b
                b = a + delta * np.abs(c - a)
            else:
                opt_score = score_d
                opt_bw = d
                a = b
                b = d
                d = c - delta * np.abs(c - a)

            diff = score_b - score_d
            score = opt_score
        return opt_bw

    def mpi_sgwr_fit(self, y, X, bw):
        if self.fixed:
            bw = round(bw, 3)
        sub_RSS = 0
        sub_trS = 0
        for i in self.x_chunk:
            err2, hat = self.local_fit(i, y, X, bw)
            sub_RSS += err2
            sub_trS += hat

        RSS_list = self.comm.gather(sub_RSS, root=0)
        trS_list = self.comm.gather(sub_trS, root=0)

        if self.comm.rank == 0:
            RSS = sum(RSS_list)
            trS = sum(trS_list)
            aicc = self.n * np.log((RSS) / (self.n)) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                    self.n - trS - 2.0)

            print(f"BW: {bw:.3f} >>> AICc Score: {aicc:.3f}", flush=True)
        else:
            aicc = None

        aicc = self.comm.bcast(aicc, root=0)

        return aicc

    def fit(self, y=None, X=None):
        if self.comm.rank == 0:
            print('Bandwidth optimiztion using golden section based on AICc values...', flush=True)

        if y is None:
            y = self.y
            X = self.X

        if self.bw:
            self.mpi_sgwr_fit(y, X, self.bw)
            return

        if self.comm.rank == 0:
            self.set_search_range()

        self.minbw = self.comm.bcast(self.minbw, root=0)
        self.maxbw = self.comm.bcast(self.maxbw, root=0)

        sgwr_func = lambda bw: self.mpi_sgwr_fit(y, X, bw)

        opt_bw = self.golden_section(self.minbw, self.maxbw, sgwr_func)

        # if self.fixed:
        #     opt_bw = round(opt_bw, 2)

        self.bw = opt_bw
        if self.comm.rank == 0:
            print('')
            print(f'Optimal BW: {self.bw:.3f}', flush=True)

        ### for adaptive bisquare + gaussian
        parser_arg = self.parser.parse_args()
        if parser_arg.biga:
            self.kernel = 'fixed'
            self.fixed = True
            self.adaptiveBG = True

        ### for alpha optimization call
        self.optimal_alpha()

    def alpha_fit(self, i, bw, bt_value):
        ### Similarity weight matrix
        data_matrix = self.data[self.columns].values
        data_point = data_matrix[i, :]
        diff = np.abs(data_matrix - data_point)
        combined = np.mean(diff, axis=1)
        ws = np.exp(-combined ** 2)
        ws = ws.reshape(-1, 1)

        ### Geographically weighted
        dist = cdist([self.coords[i]], self.coords).reshape(-1)

        # fixed gaussian
        if self.fixed:
            # wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
            if self.adaptiveBG:
                eps = 1e-10  # small value to prevent division by zero
                bw = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                wg = np.exp(-0.5 * (dist / (bw + eps)) ** 2).reshape(-1, 1)
            else:
                wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
        # adaptive bisquare
        else:
            maxd = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
            zs = dist / maxd
            zs[zs >= 1] = 1
            wg = ((1 - (zs) ** 2) ** 2).reshape(-1, 1)

        wi = bt_value * wg + (1 - bt_value) * ws

        X_new = self.X * np.sqrt(wi)
        Y_new = self.y * np.sqrt(wi)
        temp = np.dot(np.linalg.inv(np.dot(X_new.T, X_new)), X_new.T)
        hat = np.dot(X_new[i], temp[:, i])
        yhat = np.sum(np.dot(X_new, temp[:, i]).reshape(-1, 1) * Y_new)
        err = Y_new[i][0] - yhat

        return err * err, hat

    def alpha_optimization(self, bw, bt_value):
        sub_RSS = 0
        sub_trS = 0
        for i in self.x_chunk:
            err2, hat = self.alpha_fit(i, bw, bt_value)
            sub_RSS += err2
            sub_trS += hat

        RSS_list = self.comm.gather(sub_RSS, root=0)
        trS_list = self.comm.gather(sub_trS, root=0)

        if self.comm.rank == 0:
            RSS = sum(RSS_list)
            trS = sum(trS_list)

            aicc = self.n * np.log((RSS) / (self.n)) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                    self.n - trS - 2.0)

        else:
            aicc = None

        aicc = self.comm.bcast(aicc, root=0)

        return aicc

    ## for plotting
    def plot_alpha_curve(self, alpha_scores):

        sorted_items = sorted(alpha_scores.items())
        alphas = [item[0] for item in sorted_items]
        aiccs = [item[1] for item in sorted_items]

        plt.figure(figsize=(5, 4))
        plt.plot(alphas, aiccs, marker='o', markersize=3, linewidth=1, color='black')
        plt.xlabel('Alpha Values', fontsize=12)
        plt.ylabel('AICc', fontsize=12)
        plt.grid(False)
        plt.tight_layout()

        # Save in input directory
        input_dir = os.path.dirname(self.fname)
        output_path = os.path.join(input_dir, 'α_AICc_curve.png')
        plt.savefig(output_path, dpi=600)
        plt.close()

    def optimal_alpha(self):
        data = np.genfromtxt(self.fname, dtype=str, delimiter=',')
        ####### this is when we consider only the predictor variables
        # Get header
        header = data[0, 3:]

        self.columns = header.tolist()
        data = data[1:, 3:]
        data = [[float(value) for value in row] for row in data]

        self.data = pd.DataFrame(data, columns=self.columns)
        ##########
        self.data = self.comm.bcast(self.data, root=0)
        self.columns = self.comm.bcast(self.columns, root=0)

        if self.gwr_model:
            if self.comm.rank == 0:
                print("(alpha=1)=GWR Model", flush=True)
            self.alphatype = True
            best_alpha = 1
            self.best_alpha = self.comm.bcast(best_alpha, root=0)

        elif self.alphacurve:
            self.alphatype = False
            initial_candidates = [0.9, 0.01]
            alpha_scores = {}
            best_alpha = None
            best_score = float('inf')

            def evaluate_alpha(alpha):
                alpha = self.comm.bcast(alpha if self.comm.rank == 0 else None, root=0)
                aicc = self.alpha_optimization(self.bw, alpha)
                aicc = self.comm.bcast(aicc, root=0)
                return float(aicc)

            # Step 1: Evaluate initial candidates
            for alpha in initial_candidates:
                if alpha < 0.02:
                    continue
                aicc = evaluate_alpha(alpha)
                if self.comm.rank == 0:
                    alpha_scores[alpha] = aicc
                    print(f"Alpha: {alpha:.3f} >>> AICc Score: {aicc:.3f}", flush=True)
                    if aicc < best_score:
                        best_score = aicc
                        best_alpha = alpha

            # Step 2: Broadcast best_alpha and best_score
            best_alpha = self.comm.bcast(best_alpha if self.comm.rank == 0 else None, root=0)
            best_score = self.comm.bcast(best_score if self.comm.rank == 0 else None, root=0)

            def recursive_search(low, high, depth=0):
                if abs(high - low) < 0.01 or depth > 10:
                    return

                mid = round((low + high) / 2, 3)
                if mid in alpha_scores or mid < 0.02:
                    return

                aicc = self.alpha_optimization(self.bw, mid)
                alpha_scores[mid] = aicc

                nonlocal best_alpha, best_score
                if self.comm.rank == 0:
                    print(f'Alpha: {mid:.3f} >>> AICc Score: {aicc:.3f}', flush=True)
                if aicc < best_score:
                    best_alpha = mid
                    best_score = aicc
                    # Keep searching both sides around this new best
                    recursive_search(low, mid, depth + 1)
                    recursive_search(mid, high, depth + 1)
                else:
                    # Only search between previous best and mid
                    if mid < best_alpha:
                        recursive_search(mid, best_alpha, depth + 1)
                    else:
                        recursive_search(best_alpha, mid, depth + 1)

            # Start search around neighbors
            sorted_initial = sorted(initial_candidates)
            best_index = sorted_initial.index(best_alpha)
            if best_index > 0:
                recursive_search(sorted_initial[best_index - 1], best_alpha)
            if best_index < len(sorted_initial) - 1:
                recursive_search(best_alpha, sorted_initial[best_index + 1])

            # Beyond 0.9 toward 1
            if best_alpha >= 0.9:
                upper_bound = 1.0
                step_alpha = 0.05
                current = best_alpha

                while current < upper_bound:
                    next_alpha = round(current + step_alpha, 3)
                    if next_alpha >= upper_bound:
                        next_alpha = upper_bound

                    if next_alpha in alpha_scores:
                        break

                    aicc = self.alpha_optimization(self.bw, next_alpha)
                    alpha_scores[next_alpha] = aicc
                    if self.comm.rank == 0:
                        print(f'Alpha: {next_alpha:.3f} >>> AICc Score: {aicc:.3f}', flush=True)

                    if aicc < best_score:
                        best_alpha = next_alpha
                        best_score = aicc
                        recursive_search(current, next_alpha)
                        current = next_alpha
                    else:
                        break

                    # Stop if we’ve reached the top
                    if next_alpha == upper_bound:
                        break

            self.best_alpha = self.comm.bcast(best_alpha, root=0)

            if self.comm.rank == 0:
                input_dir = os.path.dirname(self.fname)
                output_path = os.path.join(input_dir, 'alpha_scores.csv')
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['alpha', 'aicc'])
                    for alpha, aicc in sorted(alpha_scores.items()):
                        writer.writerow([alpha, aicc])
                self.plot_alpha_curve(alpha_scores)
        else:
            self.alphatype = False

            def evaluate_alpha(alpha):
                alpha = self.comm.bcast(alpha, root=0)
                aicc = self.alpha_optimization(self.bw, alpha)
                aicc = self.comm.bcast(aicc, root=0)
                return float(aicc)

            initial_candidates = [0.7, 0.5, 0.1]
            alpha_scores = {}
            best_alpha = None
            best_score = float("inf")
            stop_evaluation = False

            for alpha in initial_candidates:
                if alpha < 0.01 or alpha > 0.9:
                    continue
                aicc = evaluate_alpha(alpha)
                if self.comm.rank == 0:
                    alpha_scores[alpha] = aicc
                    print(f"Alpha: {alpha:.3f} >>> AICc Score: {aicc:.3f}", flush=True)
                    if aicc < best_score:
                        best_score = aicc
                        best_alpha = alpha
                        stop_evaluation = False
                    else:
                        stop_evaluation = True

                stop_evaluation = self.comm.bcast(stop_evaluation, root=0)

                if stop_evaluation:
                    break

            def greedy_direction(start, direction):
                nonlocal best_alpha, best_score
                current = start

                while True:
                    step = 0.1 if current > 0.1 else 0.02
                    next_alpha = round(current + direction * step, 3)

                    # All ranks check if they should exit
                    stop_condition = (
                            next_alpha < 0.01 or next_alpha > 0.9 or next_alpha in alpha_scores
                    )
                    stop_condition = self.comm.bcast(stop_condition, root=0)
                    if stop_condition:
                        break

                    # All evaluate the same alpha
                    aicc = evaluate_alpha(next_alpha)

                    # Rank 0 decides what to do
                    if self.comm.rank == 0:
                        alpha_scores[next_alpha] = aicc
                        print(f"Alpha: {next_alpha:.3f} >>> AICc Score: {aicc:.3f}", flush=True)
                        if aicc < best_score:
                            best_score = aicc
                            best_alpha = next_alpha
                            current = next_alpha
                            stop = False
                        else:
                            stop = True
                    else:
                        stop = None
                        current = next_alpha  # Keep direction in sync

                    stop = self.comm.bcast(stop, root=0)

                    if stop:
                        break

            best_alpha = self.comm.bcast(best_alpha if self.comm.rank == 0 else None, root=0)
            greedy_direction(best_alpha, -1)  # Explore lower alphas
            greedy_direction(best_alpha, 1)  # Explore higher alphas

            # beyond 0.9
            do_extended = self.comm.bcast(best_alpha >= 0.9 if self.comm.rank == 0 else None, root=0)
            if do_extended:
                upper_bound = 1.0
                step_alpha = 0.05
                current = best_alpha

                while current < upper_bound:
                    next_alpha = round(current + step_alpha, 3)
                    if next_alpha > upper_bound:
                        next_alpha = upper_bound

                    # Synchronize next_alpha across all ranks
                    next_alpha = self.comm.bcast(next_alpha if self.comm.rank == 0 else None, root=0)

                    # All ranks evaluate
                    aicc = self.alpha_optimization(self.bw, next_alpha)
                    aicc = self.comm.bcast(aicc, root=0)

                    if self.comm.rank == 0:
                        alpha_scores[next_alpha] = aicc
                        print(f"Alpha: {next_alpha:.3f} >>> AICc Score: {aicc:.3f}", flush=True)
                        improved = aicc < best_score
                        if improved:
                            best_alpha = next_alpha
                            best_score = aicc
                        else:
                            improved = False
                    else:
                        improved = None

                    # Synchronize stopping condition
                    improved = self.comm.bcast(improved, root=0)
                    if not improved:
                        break

                    current = next_alpha

            if self.comm.rank == 0:
                input_dir = os.path.dirname(self.fname)
                output_path = os.path.join(input_dir, 'alpha_scores.csv')

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['alpha', 'aicc'])
                    for alpha, aicc in sorted(alpha_scores.items()):
                        writer.writerow([alpha, aicc])

                self.plot_alpha_curve(alpha_scores)

            self.best_alpha = self.comm.bcast(best_alpha, root=0)

        if self.comm.rank == 0:
            print()
            print('Model is fitting with bw: {} and alpha: {}'.format(self.bw, self.best_alpha), flush=True)

        final_model = ALPHA_OPT(self.coords, self.y, self.X, self.bw, self.data, self.columns, self.comm, self.iter,
                                self.best_alpha, self.fname, self.alphatype, self.adaptiveBG, kernel=self.kernel,
                                fixed=self.fixed).alpha_opt_mpi()

        if self.comm.rank == 0:
            print('Running time:', datetime.datetime.now() - self.start_time)


