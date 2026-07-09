import math

class GaussianMixture1D:
    def __init__(self, data, mu1, mu2, var1, var2, pi1=0.5, pi2=0.5):
        self.data = data
        self.N = len(data)
        
        # Initial Parameters: Cluster 1 (Children), Cluster 2 (Pros)
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.var1 = float(var1)
        self.var2 = float(var2)
        self.pi1 = float(pi1)
        self.pi2 = float(pi2)
        
        # Array to store responsibilities (gamma_i1)
        self.gammas = [0.0] * self.N

    @staticmethod
    def _gaussian_pdf(x, mu, var):
        """Helper method to calculate normal distribution probability density."""
        if var <= 0:
            var = 1e-6 # Guard against division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = -((x - mu) ** 2) / (2.0 * var)
        return coeff * math.exp(exponent)

    def calculate_log_likelihood(self):
        """Compute current Log-Likelihood of the dataset given parameters."""
        log_l = 0.0
        for x in self.data:
            p_x = (self.pi1 * self._gaussian_pdf(x, self.mu1, self.var1) + 
                   self.pi2 * self._gaussian_pdf(x, self.mu2, self.var2))
            log_l += math.log(max(p_x, 1e-12)) # Guard against log(0)
        return log_l

    def e_step(self):
        """Expectation Step: Calculate responsibilities (posterior probabilities)."""
        for i, x in enumerate(self.data):
            p1 = self.pi1 * self._gaussian_pdf(x, self.mu1, self.var1)
            p2 = self.pi2 * self._gaussian_pdf(x, self.mu2, self.var2)
            total = p1 + p2
            # Responsibility that point i belongs to cluster 1 (Children)
            self.gammas[i] = p1 / total if total > 0 else 0.5

    def m_step(self):
        """Maximization Step: Update parameters using responsibilities."""
        # Calculate effective number of points in each cluster
        N1 = sum(self.gammas)
        N2 = self.N - N1
        
        if N1 == 0 or N2 == 0:
            return 
        
        # Update Mixing Coefficients (pi)
        self.pi1 = N1 / self.N
        self.pi2 = N2 / self.N
        
        # Update Means (mu)
        self.mu1 = sum(g * x for g, x in zip(self.gammas, self.data)) / N1
        self.mu2 = sum((1.0 - g) * x for g, x in zip(self.gammas, self.data)) / N2
        
        # Update Variances (sigma^2)compute_cost
        self.var1 = sum(g * ((x - self.mu1) ** 2) for g, x in zip(self.gammas, self.data)) / N1
        self.var2 = sum((1.0 - g) * ((x - self.mu2) ** 2) for g, x in zip(self.gammas, self.data)) / N2

    def print_tracking_row(self, iteration):
        """Rubric Requirement: Print tracking metrics formatted nicely."""
        log_l = self.calculate_log_likelihood()
        print(f"{iteration:11d} | {self.mu1:8.2f} | {self.mu2:8.2f} | {self.var1:8.2f} | {self.var2:8.2f} | {self.pi1:6.3f} | {self.pi2:6.3f} | {log_l:12.4f}")

    def fit(self, max_iterations=20, tolerance=1e-6):
        """Run EM loop and output the exact tracking table for the presentation deck."""
        print("-" * 88)
        print("Iteration   | μ1 (Child)| μ2 (Pro)  | σ1²       | σ2²       | π1     | π2     | Log-Likelihood")
        print("-" * 88)
        
        # Print Iteration 0 (Initialization)
        self.print_tracking_row(0)
        
        prev_log_l = self.calculate_log_likelihood()
        
        for iteration in range(1, max_iterations + 1):
            self.e_step()
            self.m_step()
            
            # Print Iteration 1 and 2 explicitly for rubric tracking table
            if iteration <= 2 or iteration == max_iterations:
                self.print_tracking_row(iteration)
                
            current_log_l = self.calculate_log_likelihood()
            if abs(current_log_l - prev_log_l) < tolerance:
                print(f"\nConverged successfully at iteration {iteration}!")
                break
            prev_log_l = current_log_l

    def predict_posterior(self, test_height):
        """Live Presentation Requirement: Classify a random test height from coach."""
        p1 = self.pi1 * self._gaussian_pdf(test_height, self.mu1, self.var1)
        p2 = self.pi2 * self._gaussian_pdf(test_height, self.mu2, self.var2)
        total = p1 + p2
        
        prob_child = (p1 / total) * 100 if total > 0 else 0.0
        prob_pro = (p2 / total) * 100 if total > 0 else 0.0
        
        print(f"\n--- Live Classification for Height: {test_height} inches ---> {test_height * 2.54} cm")
        print(f"Probability it is a Child:             {prob_child:.2f}%")
        print(f"Probability it is a Basketball Player: {prob_pro:.2f}%")
        return prob_child, prob_pro