""" Related to alpha optimization in SGWR model"""
""" Author: M. Naser Lessani. email:naserlessani252@gamil.com"""

import numpy as np
from spglm.family import Gaussian
from scipy.spatial.distance import cdist

class ALPHA(object):
    def __init__(self, coords, y, X, data, family=Gaussian(),
                 offset=None, kernel='bisquare', bw=False, fixed=False, greedy=False, adaptiveBG=False, constant=True):

        self.coords = coords
        self.y = y
        self.X = X
        self.data = X
        self.family = family
        self.offset = offset
        self.kernel = kernel
        self.fixed = fixed
        self.bw = bw
        self.data = data
        self.alphacurve = greedy
        self.constant = constant
        self.n = self.X.shape[0]
        self.k = self.X.shape[1]
        self.minbw = None
        self.maxbw = None
        self.adaptiveBG = adaptiveBG
        self.search_params = {}

        if self.adaptiveBG:
            self.fixed = False
        if self.constant:
            self.X = np.hstack([np.ones((self.n, 1)), self.X])  ## adding the intercept

    ########
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

    def sgwr_fit(self, y, X, bw):
        if self.fixed:
            bw = round(bw, 3)
        RSS = 0
        trS = 0
        for i in range(self.n):
            err2, hat = self._local_fit(i, y, X, bw)
            RSS += err2
            trS += hat

        aicc = self.n * np.log((RSS) / (self.n)) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                self.n - trS - 2.0)

        print(f"BW: {bw:.3f} >>> AICc Score: {aicc:.3f}", flush=True)

        return aicc

    def fit(self, y=None, X=None):
        if self.bw:  ## if the bw is given by the users
            alpha_optimization = self.optimal_alpha()
        else:  ## if the model requires to optimize the bandwidth
            print('Bandwidth optimization using golden section based on AICc values...', flush=True)
            if y is None:
                y = self.y
                X = self.X

            self.set_search_range()

            sgwr_func = lambda bw: self.sgwr_fit(y, X, bw)

            self.bw = self.golden_section(self.minbw, self.maxbw, sgwr_func)
            print('')
            print(f'Optimal BW: {self.bw:.3f}')
            # Save search params for spatial_variability compatibility
            # return round(self.bw, 3)
            if self.adaptiveBG:
                self.fixed = True
            alpha_optimization = self.optimal_alpha()

            return alpha_optimization

    def fit(self, y=None, X=None):
        # If user provides a fixed bandwidth, skip optimization
        if self.bw:
            alpha_optimization = self.optimal_alpha()
        else:
            print('Bandwidth optimization using golden section based on AICc values...', flush=True)

            if y is None:
                y = self.y
                X = self.X

            self.set_search_range()
            sgwr_func = lambda bw: self.sgwr_fit(y, X, bw)

            self.bw = self.golden_section(self.minbw, self.maxbw, sgwr_func)

            print(f'\nOptimal BW: {self.bw:.3f}')

            if self.adaptiveBG:
                self.fixed = True

            alpha_optimization = self.optimal_alpha()

        return alpha_optimization

    def alpha_fit(self, i, bw, bt_value):
        ###
        data_point = self.data.iloc[i, :].values  # fix: use .iloc and convert to np array
        diff = np.abs(self.data.values - data_point)
        combined = np.mean(diff, axis=1)
        ws = np.exp(-combined ** 2)
        ws = ws.reshape(-1, 1)

        ### Geographically weighted
        dist = cdist([self.coords[i]], self.coords).reshape(-1)

        # fixed gaussian
        if self.fixed:
            if self.adaptiveBG:
                bw = np.partition(dist, int(bw) - 1)[int(bw) - 1] * 1.0000001
                wg = np.exp(-0.5 * (dist / bw) ** 2).reshape(-1, 1)
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
        RSS = 0
        trS = 0
        for i in range(self.n):
            err2, hat = self.alpha_fit(i, bw, bt_value)
            RSS += err2
            trS += hat

        aicc = self.n * np.log((RSS) / (self.n)) + self.n * np.log(2 * np.pi) + self.n * (self.n + trS) / (
                self.n - trS - 2.0)

        return [aicc]

    def greedy_fit(self):
        print('')
        print('Greedy optimization takes longer time than divide and conqur option!')
        print('Alpha optimization using bandwidth:', round(self.bw, 3))
        print('')
        initial_candidates = [0.9, 0.01]
        alpha_scores = {}
        best_alpha = None
        best_score = float('inf')

        # Evaluate initial candidates
        for alpha in initial_candidates:
            if alpha < 0.02:
                continue
            aicc = self.alpha_optimization(self.bw, alpha)[0]
            alpha_scores[alpha] = aicc
            print(f'Alpha: {alpha:.3f} >>> AICc Score: {aicc:.3f}', flush=True)
            if aicc < best_score:
                best_score = aicc
                best_alpha = alpha
            else:
                break

        # Recursive search between best and neighbors
        def recursive_search(low, high, depth=0):
            if abs(high - low) < 0.01 or depth > 10:
                return

            mid = round((low + high) / 2, 3)
            if mid in alpha_scores or mid < 0.02:
                return

            aicc = self.alpha_optimization(self.bw, mid)[0]
            alpha_scores[mid] = aicc
            print(f'Alpha: {mid:.3f} >>> AICc Score: {aicc:.3f}', flush=True)

            nonlocal best_alpha, best_score
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

        return best_alpha

    ### this is coarse divide-and-conquer strategy with a step size of 0.1
    def divid_fit(self):
        print('')
        print('Alpha optimization using bandwidth:', round(self.bw, 3))
        print('')
        initial_candidates = [0.7, 0.5, 0.1]
        alpha_scores = {}
        best_alpha = None
        best_score = float('inf')

        # Step 1: Evaluate initial candidates
        for alpha in initial_candidates:
            aicc = self.alpha_optimization(self.bw, alpha)[0]
            alpha_scores[alpha] = aicc
            print(f'Alpha: {alpha:.3f} >>> AICc Score: {aicc:.3f}', flush=True)
            if aicc < best_score:
                best_score = aicc
                best_alpha = alpha
            else:
                break

        # Step 2: Greedy search (strictly stop if score worsens)
        def greedy_direction_search(start, direction, min_alpha=0.01):
            nonlocal best_alpha, best_score

            current = start
            while True:
                step = 0.1 if current > 0.1 else 0.02
                next_alpha = round(current + direction * step, 3)
                if next_alpha < min_alpha or next_alpha > 1.0 or next_alpha in alpha_scores:
                    break

                aicc = self.alpha_optimization(self.bw, next_alpha)[0]
                alpha_scores[next_alpha] = aicc
                print(f'Alpha: {next_alpha:.3f} >>> AICc Score: {aicc:.3f}', flush=True)

                if aicc < best_score:
                    best_alpha = next_alpha
                    best_score = aicc
                    current = next_alpha
                else:
                    break

        # Step 3: Explore both directions around best_alpha
        greedy_direction_search(best_alpha, direction=-1)  # Lower alphas
        greedy_direction_search(best_alpha, direction=1)  # Higher alphas

        return best_alpha

    def optimal_alpha(self):
        if self.alphacurve:
            self.best_alpha = self.greedy_fit()
            print('')
            print('Best alpha value (greedy):', self.best_alpha)
            return round(self.bw, 3), self.best_alpha

        else:
            self.best_alpha = self.divid_fit()
            print('')
            print('Best alpha value (divide and conquer):', self.best_alpha)
            return round(self.bw, 3), self.best_alpha